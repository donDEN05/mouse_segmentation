"""
Скрипт для проверки всех зависимостей и настроек перед запуском
prepare_dataset.py, yolo_fine_tuning.py и eda.ipynb
"""
import sys
import importlib
from pathlib import Path


def check_python_version():
    """Проверка версии Python"""
    print("=" * 60)
    print("ПРОВЕРКА ВЕРСИИ PYTHON")
    print("=" * 60)
    version = sys.version_info
    print(f"Python версия: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ ОШИБКА: Требуется Python 3.8 или выше")
        return False
    else:
        print("✓ Версия Python подходит")
        return True


def check_package(package_name, import_name=None, required=True):
    """
    Проверка наличия пакета
    
    Parameters:
    -----------
    package_name : str
        Имя пакета для отображения
    import_name : str
        Имя для импорта (если отличается от package_name)
    required : bool
        Обязателен ли пакет
        
    Returns:
    --------
    bool : True если пакет установлен, False иначе
    """
    import_name = import_name or package_name
    try:
        importlib.import_module(import_name)
        print(f"✓ {package_name} установлен")
        return True
    except ImportError:
        if required:
            print(f"❌ {package_name} НЕ УСТАНОВЛЕН (обязательный)")
        else:
            print(f"⚠ {package_name} не установлен (опциональный)")
        return False


def check_cuda():
    """Проверка наличия CUDA и PyTorch с поддержкой CUDA"""
    print("\n" + "=" * 60)
    print("ПРОВЕРКА CUDA")
    print("=" * 60)
    
    try:
        import torch
        print(f"✓ PyTorch установлен (версия: {torch.__version__})")
        
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            print(f"✓ CUDA доступна")
            print(f"  - Количество GPU: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                print(f"  - GPU {i}: {gpu_name}")
                print(f"    CUDA версия: {torch.version.cuda}")
        else:
            print("⚠ CUDA недоступна (будет использоваться CPU)")
            print("  Для ускорения обучения рекомендуется использовать GPU")
        
        return cuda_available
    except ImportError:
        print("❌ PyTorch не установлен")
        return False


def check_file_exists(file_path, description):
    """
    Проверка существования файла
    
    Parameters:
    -----------
    file_path : str
        Путь к файлу
    description : str
        Описание файла
        
    Returns:
    --------
    bool : True если файл существует, False иначе
    """
    path = Path(file_path)
    if path.exists():
        print(f"✓ {description}: {file_path}")
        return True
    else:
        print(f"❌ {description} не найден: {file_path}")
        return False


def main():
    """Основная функция проверки зависимостей"""
    print("\n" + "=" * 60)
    print("ПРОВЕРКА ЗАВИСИМОСТЕЙ ДЛЯ ПРОЕКТА")
    print("=" * 60)
    
    all_ok = True
    
    # Проверка версии Python
    all_ok = check_python_version() and all_ok
    
    # Общие зависимости (для всех скриптов)
    print("\n" + "=" * 60)
    print("ОБЩИЕ ЗАВИСИМОСТИ")
    print("=" * 60)
    
    common_packages = [
        ("numpy", None, True),
        ("pandas", None, True),
        ("pathlib", None, True),
    ]
    
    for package, import_name, required in common_packages:
        if not check_package(package, import_name, required):
            if required:
                all_ok = False
    
    # Зависимости для prepare_dataset.py
    print("\n" + "=" * 60)
    print("ЗАВИСИМОСТИ ДЛЯ prepare_dataset.py")
    print("=" * 60)
    
    prepare_packages = [
        ("json", None, True),
        ("cv2", "cv2", True),
        ("sklearn", "sklearn.model_selection", True),
    ]
    
    for package, import_name, required in prepare_packages:
        if not check_package(package, import_name, required):
            if required:
                all_ok = False
    
    # Зависимости для yolo_fine_tuning.py
    print("\n" + "=" * 60)
    print("ЗАВИСИМОСТИ ДЛЯ yolo_fine_tuning.py")
    print("=" * 60)
    
    yolo_packages = [
        ("ultralytics", None, True),
        ("torch", None, True),
        ("mlflow", None, True),  # Обязательный для yolo_fine_tuning.py
    ]
    
    for package, import_name, required in yolo_packages:
        if not check_package(package, import_name, required):
            if required:
                all_ok = False
    
    # Зависимости для eda.ipynb
    print("\n" + "=" * 60)
    print("ЗАВИСИМОСТИ ДЛЯ eda.ipynb")
    print("=" * 60)
    
    eda_packages = [
        ("zipfile", None, True),
        ("json", None, True),
        ("pandas", None, True),
        ("numpy", None, True),
        ("seaborn", None, True),
        ("matplotlib", None, True),
        ("cv2", "cv2", True),
        ("pathlib", None, True),
    ]
    
    for package, import_name, required in eda_packages:
        if not check_package(package, import_name, required):
            if required:
                all_ok = False
    
    # Проверка CUDA
    cuda_available = check_cuda()
    
    # Проверка наличия необходимых файлов
    print("\n" + "=" * 60)
    print("ПРОВЕРКА ФАЙЛОВ")
    print("=" * 60)
    
    required_files = [
        ("prepare_dataset.py", "Скрипт подготовки датасета"),
        ("yolo_fine_tuning.py", "Скрипт дообучения YOLO"),
        ("epm1_dataset.yaml", "Конфигурация датасета"),
        ("eda.ipynb", "Notebook для EDA"),
    ]
    
    files_ok = True
    for file_path, description in required_files:
        if not check_file_exists(file_path, description):
            files_ok = False
    
    all_ok = all_ok and files_ok
    
    # Проверка опциональных файлов/директорий
    print("\n" + "=" * 60)
    print("ОПЦИОНАЛЬНЫЕ ПРОВЕРКИ")
    print("=" * 60)
    
    optional_files = [
        ("tracking_datasets_extracted", "Разархивированный датасет"),
        ("DAMM_mouse_detection/tracking_datasets.zip", "Архив датасета"),
    ]
    
    for file_path, description in optional_files:
        check_file_exists(file_path, description)
    
    # Итоговый результат
    print("\n" + "=" * 60)
    print("ИТОГОВЫЙ РЕЗУЛЬТАТ")
    print("=" * 60)
    
    if all_ok:
        print("✓ Все обязательные зависимости установлены")
        if cuda_available:
            print("✓ CUDA доступна - обучение будет ускорено")
        else:
            print("⚠ CUDA недоступна - обучение будет медленным на CPU")
        print("\nМожно запускать скрипты!")
        return 0
    else:
        print("❌ Обнаружены проблемы с зависимостями")
        print("\nУстановите недостающие пакеты:")
        print("  pip install -r requirements.txt")
        print("\nИли установите пакеты вручную:")
        print("  pip install ultralytics torch torchvision opencv-python")
        print("  pip install pandas numpy seaborn matplotlib scikit-learn")
        print("  pip install mlflow  # опционально, но рекомендуется")
        return 1


if __name__ == "__main__":
    sys.exit(main())

