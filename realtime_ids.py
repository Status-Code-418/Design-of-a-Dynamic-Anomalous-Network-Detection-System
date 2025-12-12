# ==============================================================================
# 실시간 네트워크 침입 탐지 시스템 (IDS) - SYN 패킷 강제 ALERT 및 IPTABLES 차단 로직 적용
# ==============================================================================
import joblib
import pandas as pd
import numpy as np
import sys
import warnings
import datetime
import subprocess  # iptables 명령 실행을 위한 subprocess 모듈
from sklearn.preprocessing import LabelEncoder
from collections import Counter

# Scapy는 NIC 접근을 위해 sudo 권한이 필요
try:
    from scapy.all import sniff, IP, TCP, UDP, ICMP
except ImportError:
    print("Error: Scapy library not found. Please run 'sudo pip3 install scapy'.")
    sys.exit(1)

# 경고 무시 설정 (모델 예측 시 발생하는 경고를 숨기기 위함)
warnings.filterwarnings('ignore')

# --- 설정 (Config) ---
INTERFACE = "eth0"  # 패킷을 캡처할 NIC 이름 (예: eth0, ens33). 필요에 따라 변경
CAPTURE_COUNT = 100 # 한 번에 캡처할 패킷 수
MODEL_TYPE = "RandomForestClassifier" # 사용할 모델 유형 (HistGradientBoostingClassifier 또는 RandomForestClassifier)
DATASET_TYPE = "full" if MODEL_TYPE in ["RandomForestClassifier", "HistGradientBoostingClassifier"] else "reduced"
LOG_FILE_PATH = "ids_detection_log.txt" # 로그 파일 경로
PROBE_THRESHOLD = 0.05
# IPTABLES 차단 설정
ENABLE_IPTABLES_BLOCKING = True # IPTABLES 차단 기능 활성화/비활성화 (True: 활성화, False: 비활성화)
BLOCK_ON_DEBUG_ALERT = True # 디버그 SYN ALERT에도 차단 적용 여부 (False로 설정하면 디버그 ALERT는 차단하지 않음)

# --- 전역 변수 로드 (Load Global Components) ---
try:
    best_model = joblib.load('nsl_kdd_best_model.pkl')
    attack_category_encoder = joblib.load('nsl_kdd_label_encoder.pkl')
    protocol_le = joblib.load('nsl_kdd_protocol_le.pkl')
    service_le = joblib.load('nsl_kdd_service_le.pkl')
    flag_le = joblib.load('nsl_kdd_flag_le.pkl')

    if DATASET_TYPE == "full":
        feature_cols = joblib.load('nsl_kdd_full_features.pkl')
        scaler = None
    else: # reduced
        feature_cols = joblib.load('nsl_kdd_reduced_features.pkl')
        scaler = joblib.load('nsl_kdd_scaler.pkl')
    
    class_labels = attack_category_encoder.classes_
    print("Model and Preprocessing components loaded successfully.")
    print(f"Model: {MODEL_TYPE}, Dataset: {DATASET_TYPE}, Features: {len(feature_cols)}")
except FileNotFoundError as e:
    print(f"\nERROR: One or more required model/preprocessing files not found: {e}")
    print("   Please run 'nsl_kdd_project.py' first to train and save the components.")
    sys.exit(1)

# 로그 파일 설정 및 시작 메시지 기록
log_file = None # 전역 변수로 선언
try:
    log_file = open(LOG_FILE_PATH, "a")
    log_file.write(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] IDS Session Started. Logging to {LOG_FILE_PATH}\n")
    print(f"Logging started. Results will be saved to {LOG_FILE_PATH}")
    if ENABLE_IPTABLES_BLOCKING:
        print(f"IPTABLES blocking is ENABLED. Malicious IPs will be blocked.")
        if BLOCK_ON_DEBUG_ALERT:
            print(f"DEBUG MODE ACTIVE: All TCP SYN packets will be marked as Probe(DEBUG) AND WILL BE BLOCKED.")
        else:
            print(f"DEBUG MODE ACTIVE: All TCP SYN packets will be marked as Probe(DEBUG) but WILL NOT BE BLOCKED (BLOCK_ON_DEBUG_ALERT=False).")
    else:
        print(f"IPTABLES blocking is DISABLED. Detections will only be logged.")

except Exception as e:
    print(f"FATAL ERROR: Could not open log file {LOG_FILE_PATH}: {e}")
    sys.exit(1)

# 이미 차단된 IP 주소를 저장하는 집합 (중복 차단 방지)
blocked_ips = set()

# --- IPTABLES 차단 규칙 추가 함수 ---
def add_iptables_block_rule(ip_address, log_file):
    """지정된 IP 주소를 iptables로 차단하는 규칙"""
    if not ENABLE_IPTABLES_BLOCKING:
        return False

    if ip_address in blocked_ips:
        return False

    try:
        # INPUT 체인에 차단 규칙 (해당 서버로 들어오는 패킷 차단)
        # FORWARD 체인에 차단 규칙 (해당 서버를 경유하는 패킷 차단, 라우터 역할을 할 경우)
        command_input = ["sudo", "iptables", "-A", "INPUT", "-s", ip_address, "-j", "DROP"]
        command_forward = ["sudo", "iptables", "-A", "FORWARD", "-s", ip_address, "-j", "DROP"]
        
        result_input = subprocess.run(command_input, capture_output=True, text=True, check=True)
        result_forward = subprocess.run(command_forward, capture_output=True, text=True, check=True)

        if log_file:
            log_file.write(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}] [IPTABLES_BLOCK] Successfully blocked IP: {ip_address}\n")
        print(f"IPTABLES BLOCKED: {ip_address}")
        blocked_ips.add(ip_address)
        return True
    except subprocess.CalledProcessError as e:
        if log_file:
            log_file.write(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}] [IPTABLES_ERROR] Failed to block IP {ip_address}: {e.stderr.strip()}\n")
        print(f"IPTABLES BLOCK FAILED for {ip_address}: {e.stderr.strip()}")
        return False
    except FileNotFoundError:
        if log_file:
            log_file.write(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}] [IPTABLES_ERROR] 'iptables' command not found. Ensure iptables is installed and in PATH.\n")
        print("'iptables' command not found. Ensure iptables is installed and in PATH.")
        return False


# --- NSL-KDD Feature Mapping Function ---
def create_nsl_kdd_features(packet):
    """
    Scapy 패킷에서 NSL-KDD 데이터셋에 필요한 41개 특징 중 일부를 추출하고 기본값을 설정합니다.
    """
    features = {col: 0 for col in feature_cols}
    features["duration"] = 0
    features["src_bytes"] = len(packet)
    features["dst_bytes"] = 0

    # Protocol Type
    features["protocol_type"] = 'icmp'
    if IP in packet:
        if packet[IP].proto == 6: features["protocol_type"] = 'tcp'
        elif packet[IP].proto == 17: features["protocol_type"] = 'udp'
        elif packet[IP].proto == 1: features["protocol_type"] = 'icmp'

    # Service/Port (단순화)
    if TCP in packet:
        features["service"] = str(packet[TCP].dport)
    elif UDP in packet:
        features["service"] = str(packet[UDP].dport)
    else:
        features["service"] = 'other'

    features["flag"] = 'SF'
    if TCP in packet:
        flags = packet[TCP].flags
        if 'S' in flags and 'A' not in flags: features["flag"] = 'S0' # SYN만 있는 경우 (Initial SYN, No ACK)
        elif 'R' in flags: features["flag"] = 'REJ' # RST flag
        elif 'F' in flags and 'A' in flags: features["flag"] = 'RSTR' # FIN/ACK flag
        elif 'S' in flags and 'A' in flags: features["flag"] = 'S1' # SYN/ACK flag

    # 기타 이진 Features (항상 0으로 설정하여 단순화)
    features["land"] = 0
    features["logged_in"] = 0
    features["is_guest_login"] = 0

    # 수치적 Features (Contextual features는 0으로 단순화)
    features["count"] = 1
    
    def safe_transform(le, val, default_val=0):
        try:
            return le.transform([val])[0]
        except ValueError:
            # 학습되지 않은 레이블은 디폴트 값으로 처리
            # ex: service_le.classes_에 'http'가 없다면 0, 있다면 'http'에 해당하는 인코딩 값.
            #     'other'는 보통 항상 존재한다고 가정
            if val not in le.classes_:
                if 'other' in le.classes_:
                    return le.transform(['other'])[0]
                else:
                    return default_val
            return le.transform([val])[0]

    features_df = pd.DataFrame([features])
    features_df["protocol_type"] = safe_transform(protocol_le, features_df["protocol_type"].iloc[0])
    
    # service 컬럼의 기본값 처리를 보다 유연하게
    # NSL-KDD 데이터셋의 service에는 다양한 포트 번호가 문자열로 존재할 수 있으므로,
    # 학습 데이터에 없는 service가 들어오면 'other'로 처리
    if str(features_df["service"].iloc[0]) not in service_le.classes_:
        if 'other' in service_le.classes_:
            features_df["service"] = service_le.transform(['other'])[0]
        else: # 'other'도 없다면 임의의 기본값 사용 (데이터셋에 따라 다름)
            features_df["service"] = 0 
    else:
        features_df["service"] = service_le.transform(features_df["service"])[0]

    features_df["flag"] = safe_transform(flag_le, features_df["flag"].iloc[0])
    
    return features_df.iloc[0].to_dict()


# --- 모델 추론 함수 (Inference Function) ---
def predict_packet(packet_data):
    """주어진 패킷 데이터 (DataFrame)를 전처리하고 모델로 예측 및 확률을 반환합니다."""
    
    df = pd.DataFrame([packet_data])
    # 모델 학습 시 사용된 컬럼과 동일하게 정렬하고 없는 컬럼은 0
    df = df.reindex(columns=feature_cols, fill_value=0)
    
    # 모델에 따라 스케일링 적용
    if scaler:
        X_processed = scaler.transform(df)
    else:
        X_processed = df.values
        
    # 기본 예측은 Normal로 설정
    prediction_label = "Normal" 
    prediction_proba = np.zeros(len(class_labels)) 
    
    try:
        prediction_encoded = best_model.predict(X_processed)[0]
        prediction_label = attack_category_encoder.inverse_transform([prediction_encoded])[0]
        prediction_proba = best_model.predict_proba(X_processed)[0]
    except Exception as e:
        if log_file:
            log_file.write(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}] [MODEL_ERROR] Prediction failed: {e}\n")
        prediction_label = "ModelError" # 예측 실패 시 임시 레이블
        print(f" Model Prediction Error: {e}")
    
    return prediction_label, prediction_proba


# --- 패킷 처리 핸들러 (Packet Handler) ---
def packet_handler(packet):
    """캡처된 개별 패킷을 처리하고 예측 결과를 파일에 기록"""
    
    if IP not in packet:
        return
        
    # 1. NSL-KDD 특징 추출
    try:
        nsl_kdd_data = create_nsl_kdd_features(packet)
    except Exception as e:
        if log_file:
            log_file.write(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}] [ERROR] Feature creation failed for packet from {packet[IP].src}: {e}\n")
        return

    # 2. 모델 예측
    # prediction 변수에 모델의 실제 예측 결과를 담습니다.
    model_prediction, _ = predict_packet(nsl_kdd_data) 
    
    is_alert = False # 최종 ALERT 여부
    final_classification_label = model_prediction # 최종 분류 레이블 (기본은 모델 예측)

    # 디버깅 로직: SYN 패킷을 무조건 ALERT로 처리
    # Nmap 스캔은 보통 TCP SYN 패킷(플래그 S0)을 사용
    is_syn_debug_alert = False
    if IP in packet and TCP in packet:
        flags = packet[TCP].flags
        # SYN 패킷인 경우 (SYN set, ACK not set, NSL-KDD flag 'S0')
        if 'S' in flags and 'A' not in flags and not 'F' in flags and not 'R' in flags: 
            is_syn_debug_alert = True
            is_alert = True
            final_classification_label = "Probe(SYN_DEBUG)" # 디버그용 ALERT임을 명시

    # 모델이 Normal이 아닌 다른 것으로 분류했다면 Alert
    if model_prediction != "Normal" and model_prediction != "ModelError":
        is_alert = True
        final_classification_label = model_prediction # 모델이 예측한 공격 유형

    # 3. 결과 파일 기록 및 IPTABLES 차단
    src_ip = packet[IP].src
    dst_ip = packet[IP].dst
    # Scapy 패킷의 프로토콜 번호를 NSL-KDD 형식의 문자열로 변환
    protocol_num = packet[IP].proto
    protocol_str = {1:'icmp', 6:'tcp', 17:'udp'}.get(protocol_num, 'other')

    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    
    log_message = f"SRC:{src_ip:<15} DST:{dst_ip:<15} PROTO:{protocol_str:<4} CLASSIFICATION:{final_classification_label}"
    
    if is_alert:
        # 경고 기록
        log_file.write(f"[{timestamp}] [ALERT] {log_message}\n")
        print(f" **ALERT: {final_classification_label} Detected** | SRC: {src_ip} | DST: {dst_ip}") 

        # IPTABLES 차단 조건: ENABLE_IPTABLES_BLOCKING이 True이고,
        #                   디버그 ALERT일 경우 BLOCK_ON_DEBUG_ALERT가 True이거나
        #                   모델 예측으로 인한 실제 ALERT일 경우
        if ENABLE_IPTABLES_BLOCKING and ( (is_syn_debug_alert and BLOCK_ON_DEBUG_ALERT) or (not is_syn_debug_alert and final_classification_label != "Normal" and final_classification_label != "ModelError")):
            add_iptables_block_rule(src_ip, log_file)
    else:
        # 정상 트래픽은 [INFO]로 파일 및 화면에 기록
        log_file.write(f"[{timestamp}] [INFO] {log_message}\n")
        print(f"[INFO] {log_message}")


# --- 메인 실행 루프 (Main Loop) ---
if __name__ == "__main__":
    print(f"\n Starting Real-Time IDS on interface {INTERFACE}...")
    print(f"   Listening for {CAPTURE_COUNT} packets at a time. Press Ctrl+C to stop.")

    try:
        while True:
            sniff(iface=INTERFACE, prn=packet_handler, count=CAPTURE_COUNT, store=0, timeout=1) 
            
    except PermissionError:
        if log_file: log_file.close()
        print("\n Permission Denied. You must run this script with **sudo** privileges.")
        sys.exit(1)
    except OSError as e:
        if log_file: log_file.close()
        if "No such device" in str(e):
            print(f"\n Error: Interface '{INTERFACE}' not found. Please check your NIC name.")
        else:
            print(f"\n OS Error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        if log_file:
            log_file.write(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] IDS Session Ended.\n")
            log_file.close()
        print("\n\n IDS stopped by user. Log file closed.")
        sys.exit(0)
