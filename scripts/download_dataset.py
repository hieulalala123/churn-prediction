import shutil
from pathlib import Path

import kagglehub


def download_and_localize_dataset(target_dir="./data"):
    # 1. Tạo thư mục đích cục bộ nếu chưa tồn tại
    local_data_path = Path(target_dir)
    local_data_path.mkdir(exist_ok=True, parents=True)

    # 2. Định nghĩa đường dẫn file đích mong muốn
    target_file = local_data_path / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

    # Nếu file đã tồn tại cục bộ rồi, không cần tải lại để tiết kiệm thời gian
    if target_file.exists():
        print(f"✓ Dataset đã tồn tại cục bộ tại: {target_file.resolve()}")
        return target_file

    print("🤖 Đang tải dataset từ Kaggle về thư mục tạm...")
    # Tải về thư mục cache mặc định của kagglehub
    cache_dir = kagglehub.dataset_download("blastchar/telco-customer-churn")
    cache_path = Path(cache_dir)

    # 3. Tìm file CSV trong đống file vừa tải về
    csv_files = list(cache_path.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError("Không tìm thấy file CSV nào trong thư mục tải về từ Kaggle!")

    # 4. Copy các file CSV tìm thấy về thư mục cục bộ của dự án
    for csv_file in csv_files:
        destination = local_data_path / csv_file.name
        shutil.copy(csv_file, destination)
        print(f"✓ Đã di chuyển thành công: {csv_file.name} -> {destination.resolve()}")

    return target_file


if __name__ == "__main__":
    # Chạy hàm tải dữ liệu
    dataset_path = download_and_localize_dataset()
    print("🚀 Sẵn sàng cho các bước tiếp theo tại:", dataset_path)
