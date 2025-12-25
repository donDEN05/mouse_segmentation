"""
Скрипт для запуска локального MLflow сервера
"""
import os
import sys
import subprocess
import argparse
from pathlib import Path


def check_mlflow_installed():
    """Проверка установлен ли MLflow"""
    try:
        import mlflow
        print(f"✓ MLflow установлен (версия: {mlflow.__version__})")
        return True
    except ImportError:
        print("❌ MLflow не установлен")
        print("Установите его через: pip install mlflow")
        return False


def start_mlflow_server(
    backend_store_uri="runs/mlflow",
    default_artifact_root=None,
    host="127.0.0.1",
    port=5000,
    workers=4
):
    """
    Запуск локального MLflow сервера
    
    Parameters:
    -----------
    backend_store_uri : str
        URI для хранения метаданных экспериментов
    default_artifact_root : str
        Путь для хранения артефактов (по умолчанию используется backend_store_uri)
    host : str
        Хост для запуска сервера
    port : int
        Порт для запуска сервера
    workers : int
        Количество воркеров
    """
    if not check_mlflow_installed():
        sys.exit(1)
    
    # Создание директории для хранения данных, если не существует
    backend_path = Path(backend_store_uri)
    backend_path.mkdir(parents=True, exist_ok=True)
    
    # Построение команды для запуска MLflow сервера
    cmd = [
        "mlflow",
        "server",
        "--backend-store-uri", str(backend_path.absolute()),
        "--host", host,
        "--port", str(port),
        "--workers", str(workers)
    ]
    
    if default_artifact_root:
        cmd.extend(["--default-artifact-root", default_artifact_root])
    else:
        # Используем backend_store_uri для артефактов, если не указан отдельный путь
        cmd.extend(["--default-artifact-root", str(backend_path.absolute())])
    
    print("=" * 60)
    print("ЗАПУСК MLFLOW СЕРВЕРА")
    print("=" * 60)
    print(f"Backend store URI: {backend_path.absolute()}")
    print(f"Host: {host}")
    print(f"Port: {port}")
    print(f"Workers: {workers}")
    print("=" * 60)
    print(f"\nMLflow UI будет доступен по адресу:")
    print(f"  http://{host}:{port}")
    print(f"\nДля использования в скриптах укажите:")
    print(f"  --mlflow_uri http://{host}:{port}")
    print("\nДля остановки сервера нажмите Ctrl+C")
    print("=" * 60 + "\n")
    
    try:
        # Запуск сервера
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n\nОстановка MLflow сервера...")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Ошибка при запуске MLflow сервера: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print("\n❌ Команда 'mlflow' не найдена")
        print("Убедитесь, что MLflow установлен: pip install mlflow")
        sys.exit(1)


def main():
    """Основная функция"""
    parser = argparse.ArgumentParser(
        description="Запуск локального MLflow сервера для отслеживания экспериментов"
    )
    parser.add_argument(
        "--backend-store-uri",
        type=str,
        default="runs/mlflow",
        help="URI для хранения метаданных экспериментов (по умолчанию: runs/mlflow)"
    )
    parser.add_argument(
        "--default-artifact-root",
        type=str,
        default=None,
        help="Путь для хранения артефактов (по умолчанию используется backend-store-uri)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Хост для запуска сервера (по умолчанию: 127.0.0.1)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Порт для запуска сервера (по умолчанию: 5000)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Количество воркеров (по умолчанию: 4)"
    )
    
    args = parser.parse_args()
    
    start_mlflow_server(
        backend_store_uri=args.backend_store_uri,
        default_artifact_root=args.default_artifact_root,
        host=args.host,
        port=args.port,
        workers=args.workers
    )


if __name__ == "__main__":
    main()

