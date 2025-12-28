"""
Скрипт для использования предобученных моделей YOLO из папки runs/detect
Позволяет выбрать модель и версию весов (best.pt или last.pt) для инференса
"""
from ultralytics import YOLO
from pathlib import Path
import argparse
from typing import Dict, Tuple, Optional


def find_available_models(runs_dir: str = "runs/detect") -> Dict[str, Dict[str, Path]]:
    """
    Сканирует папку runs/detect и находит все доступные предобученные модели
    
    Parameters:
    -----------
    runs_dir : str
        Путь к директории с результатами обучения (по умолчанию "runs/detect")
        
    Returns:
    --------
    Dict[str, Dict[str, Path]]
        Словарь с информацией о доступных моделях:
        {
            "имя_эксперимента": {
                "best": Path("путь/к/best.pt") или None,
                "last": Path("путь/к/last.pt") или None
            }
        }
    """
    runs_path = Path(runs_dir)
    
    if not runs_path.exists():
        print(f"⚠ Директория {runs_dir} не найдена")
        return {}
    
    models = {}
    
    # Проходим по всем подпапкам в runs/detect
    for experiment_dir in runs_path.iterdir():
        if not experiment_dir.is_dir():
            continue
            
        experiment_name = experiment_dir.name
        weights_dir = experiment_dir / "weights"
        
        # Проверяем наличие папки weights
        if not weights_dir.exists():
            continue
        
        # Ищем файлы best.pt и last.pt
        best_pt = weights_dir / "best.pt"
        last_pt = weights_dir / "last.pt"
        
        model_info = {
            "best": best_pt if best_pt.exists() else None,
            "last": last_pt if last_pt.exists() else None
        }
        
        # Добавляем модель только если есть хотя бы один файл весов
        if model_info["best"] or model_info["last"]:
            models[experiment_name] = model_info
    
    return models


def print_available_models(models: Dict[str, Dict[str, Path]]) -> None:
    """
    Выводит список доступных моделей в читаемом формате
    
    Parameters:
    -----------
    models : Dict[str, Dict[str, Path]]
        Словарь с информацией о доступных моделях
    """
    if not models:
        print("❌ Не найдено ни одной предобученной модели")
        print(f"   Проверьте, что в папке runs/detect есть модели с весами в папке weights/")
        return
    
    print("=" * 60)
    print("ДОСТУПНЫЕ ПРЕДОБУЧЕННЫЕ МОДЕЛИ")
    print("=" * 60)
    
    for idx, (experiment_name, model_info) in enumerate(models.items(), 1):
        print(f"\n{idx}. Эксперимент: {experiment_name}")
        
        if model_info["best"]:
            print(f"   ✓ best.pt - лучшая модель (лучшая метрика)")
        else:
            print(f"   ✗ best.pt - не найдено")
            
        if model_info["last"]:
            print(f"   ✓ last.pt - последняя модель (последняя эпоха)")
        else:
            print(f"   ✗ last.pt - не найдено")
    
    print("=" * 60)


def select_model_interactive(models: Dict[str, Dict[str, Path]]) -> Tuple[Optional[str], Optional[str]]:
    """
    Интерактивный выбор модели и версии весов
    
    Parameters:
    -----------
    models : Dict[str, Dict[str, Path]]
        Словарь с информацией о доступных моделях
        
    Returns:
    --------
    Tuple[Optional[str], Optional[str]]
        Кортеж (имя_эксперимента, версия_весов) или (None, None) если выбор отменен
    """
    if not models:
        return None, None
    
    model_list = list(models.keys())
    
    # Выбор эксперимента
    print("\nВыберите модель (введите номер):")
    try:
        choice = input(">>> ").strip()
        model_idx = int(choice) - 1
        
        if model_idx < 0 or model_idx >= len(model_list):
            print(f"❌ Неверный выбор. Введите число от 1 до {len(model_list)}")
            return None, None
        
        selected_experiment = model_list[model_idx]
        model_info = models[selected_experiment]
        
        # Проверяем, какие версии доступны
        available_versions = []
        if model_info["best"]:
            available_versions.append("best")
        if model_info["last"]:
            available_versions.append("last")
        
        if not available_versions:
            print(f"❌ В модели {selected_experiment} нет доступных весов")
            return None, None
        
        # Выбор версии весов
        print(f"\nВыберите версию весов для модели '{selected_experiment}':")
        if "best" in available_versions:
            print("  1. best.pt - лучшая модель (рекомендуется)")
        if "last" in available_versions:
            if "best" in available_versions:
                print("  2. last.pt - последняя модель")
            else:
                print("  1. last.pt - последняя модель")
        
        version_choice = input(">>> ").strip()
        
        if version_choice == "1":
            if "best" in available_versions:
                return selected_experiment, "best"
            else:
                return selected_experiment, "last"
        elif version_choice == "2" and "last" in available_versions:
            return selected_experiment, "last"
        else:
            print("❌ Неверный выбор версии")
            return None, None
            
    except ValueError:
        print("❌ Введите число")
        return None, None
    except KeyboardInterrupt:
        print("\n❌ Выбор отменен")
        return None, None


def load_model(experiment_name: str, version: str, runs_dir: str = "runs/detect") -> Optional[YOLO]:
    """
    Загружает выбранную модель YOLO
    
    Parameters:
    -----------
    experiment_name : str
        Имя эксперимента (название папки)
    version : str
        Версия весов: "best" или "last"
    runs_dir : str
        Путь к директории с результатами обучения
        
    Returns:
    --------
    Optional[YOLO]
        Загруженная модель YOLO или None в случае ошибки
    """
    model_path = Path(runs_dir) / experiment_name / "weights" / f"{version}.pt"
    
    if not model_path.exists():
        print(f"❌ Файл модели не найден: {model_path}")
        return None
    
    try:
        print(f"\nЗагрузка модели: {model_path}")
        model = YOLO(str(model_path))
        print(f"✓ Модель успешно загружена: {experiment_name}/{version}.pt")
        return model
    except Exception as e:
        print(f"❌ Ошибка при загрузке модели: {e}")
        return None


def predict_example(model: YOLO, source: str = None, device: str = None) -> None:
    """
    Пример использования модели для предсказаний
    
    Parameters:
    -----------
    model : YOLO
        Загруженная модель YOLO
    source : str, optional
        Путь к изображению, видео или директории для предсказания
        Если None, будет показан пример использования
    device : str, optional
        Устройство для инференса ("cpu", "0" для GPU 0, и т.д.)
    """
    print("\n" + "=" * 60)
    print("ПРИМЕР ИСПОЛЬЗОВАНИЯ МОДЕЛИ")
    print("=" * 60)
    
    if source is None:
        print("\nДля использования модели выполните:")
        print("\n  # Предсказание на изображении:")
        print("  results = model('path/to/image.jpg')")
        print("\n  # Предсказание на видео:")
        print("  results = model('path/to/video.mp4')")
        print("\n  # Предсказание на всех изображениях в папке:")
        print("  results = model('path/to/images/')")
        print("\n  # Сохранение результатов:")
        print("  results = model('path/to/image.jpg', save=True)")
        print("\n  # Предсказание с указанием устройства:")
        print("  results = model('path/to/image.jpg', device='cpu')")
        print("  results = model('path/to/image.jpg', device='0')  # GPU 0")
        return
    
    try:
        # Устанавливаем устройство если указано
        if device is not None:
            print(f"Использование устройства: {device}")
        
        print(f"\nВыполнение предсказания на: {source}")
        results = model(source, device=device)
        
        print("✓ Предсказание выполнено успешно")
        print(f"  Количество результатов: {len(results)}")
        
        # Выводим информацию о первом результате
        if len(results) > 0:
            result = results[0]
            if hasattr(result, 'boxes') and result.boxes is not None:
                num_detections = len(result.boxes)
                print(f"  Обнаружено объектов: {num_detections}")
                
                if num_detections > 0:
                    print("\n  Примеры обнаруженных объектов:")
                    for i, box in enumerate(result.boxes[:min(5, num_detections)]):
                        confidence = box.conf[0].item() if hasattr(box.conf, '__len__') else box.conf.item()
                        cls = int(box.cls[0].item()) if hasattr(box.cls, '__len__') else int(box.cls.item())
                        class_name = model.names[cls] if hasattr(model, 'names') else f"class_{cls}"
                        print(f"    {i+1}. {class_name}: confidence={confidence:.3f}")
        
    except Exception as e:
        print(f"❌ Ошибка при выполнении предсказания: {e}")


def get_model(experiment_name: str = None, version: str = None, runs_dir: str = "runs/detect", 
              interactive: bool = True) -> Optional[YOLO]:
    """
    Удобная функция для программного получения модели
    
    Parameters:
    -----------
    experiment_name : str, optional
        Имя модели. Если None и interactive=True, будет предложен интерактивный выбор
    version : str, optional
        Версия весов: "best" или "last". Если None, будет выбрана best (если доступна)
    runs_dir : str
        Путь к директории с результатами обучения
    interactive : bool
        Если True и experiment_name не указан, будет предложен интерактивный выбор
        
    Returns:
    --------
    Optional[YOLO]
        Загруженная модель YOLO или None в случае ошибки
        
    Examples:
    ---------
    >>> # Интерактивный выбор
    >>> model = get_model()
    
    >>> # Прямая загрузка конкретной модели
    >>> model = get_model("epm1_finetune3", "best")
    
    >>> # Автоматический выбор версии (best если доступна, иначе last)
    >>> model = get_model("epm1_finetune3")
    """
    models = find_available_models(runs_dir)
    
    if not models:
        print("❌ Нет доступных моделей")
        return None
    
    # Определение выбранной модели
    selected_experiment = experiment_name
    selected_version = version
    
    if selected_experiment is None:
        if interactive:
            selected_experiment, selected_version = select_model_interactive(models)
            if selected_experiment is None:
                return None
        else:
            print("❌ Не указано имя модели и интерактивный режим отключен")
            return None
    else:
        if selected_experiment not in models:
            print(f"❌ Модель '{selected_experiment}' не найдена")
            return None
        
        # Если версия не указана, выбираем best, если доступен, иначе last
        if selected_version is None:
            if models[selected_experiment]["best"] is not None:
                selected_version = "best"
            elif models[selected_experiment]["last"] is not None:
                selected_version = "last"
            else:
                print(f"❌ Нет доступных весов для модели '{selected_experiment}'")
                return None
        else:
            if models[selected_experiment][selected_version] is None:
                print(f"❌ Версия '{selected_version}' недоступна для модели '{selected_experiment}'")
                return None
    
    return load_model(selected_experiment, selected_version, runs_dir)


def main():
    """Основная функция для работы с предобученными моделями"""
    parser = argparse.ArgumentParser(
        description="Использование предобученных моделей YOLO из папки runs/detect",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python yolo_using.py                           # Интерактивный выбор модели
  python yolo_using.py --model epm1_finetune3 --version best
  python yolo_using.py --model epm1_finetune3 --version best --source image.jpg
  python yolo_using.py --list                    # Показать список доступных моделей
        """
    )
    
    parser.add_argument(
        "--runs-dir",
        type=str,
        default="runs/detect",
        help="Путь к директории с результатами обучения (по умолчанию: runs/detect)"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Имя модели (название эксперимента). Если не указано, будет предложен интерактивный выбор"
    )
    
    parser.add_argument(
        "--version",
        type=str,
        choices=["best", "last"],
        default=None,
        help="Версия весов: best (лучшая модель) или last (последняя модель)"
    )
    
    parser.add_argument(
        "--list",
        action="store_true",
        help="Показать список доступных моделей и выйти"
    )
    
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Путь к изображению, видео или директории для предсказания"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Устройство для инференса: 'cpu' или номер GPU (например, '0')"
    )
    
    args = parser.parse_args()
    
    # Поиск доступных моделей
    print("\n" + "=" * 60)
    print("ПОИСК ПРЕДОБУЧЕННЫХ МОДЕЛЕЙ")
    print("=" * 60)
    models = find_available_models(args.runs_dir)
    
    # Вывод списка моделей
    print_available_models(models)
    
    # Если запрошен только список, выходим
    if args.list:
        return
    
    if not models:
        print("\n❌ Нет доступных моделей для использования")
        return
    
    # Определение выбранной модели и загрузка
    model = None
    experiment_name = None
    version = None
    
    if args.model:
        # Модель указана через аргументы
        if args.model not in models:
            print(f"\n❌ Модель '{args.model}' не найдена в доступных моделях")
            print("Доступные модели:")
            for name in models.keys():
                print(f"  - {name}")
            return
        
        experiment_name = args.model
        version = args.version  # Может быть None
        
        # Загрузка модели через функцию get_model
        print("\n" + "=" * 60)
        print("ЗАГРУЗКА МОДЕЛИ")
        print("=" * 60)
        model = get_model(experiment_name, version, args.runs_dir, interactive=False)
        
        # Определяем фактически загруженную версию для вывода информации
        if model is not None:
            if version is None:
                # Определяем, какая версия была загружена
                if models[experiment_name]["best"] is not None:
                    version = "best"
                else:
                    version = "last"
    else:
        # Интерактивный выбор и загрузка
        experiment_name, version = select_model_interactive(models)
        if experiment_name is None or version is None:
            print("❌ Модель не выбрана")
            return
        
        print("\n" + "=" * 60)
        print("ЗАГРУЗКА МОДЕЛИ")
        print("=" * 60)
        model = load_model(experiment_name, version, args.runs_dir)
    
    if model is None:
        print("❌ Не удалось загрузить модель")
        return
    
    # Вывод информации о модели
    print("\n" + "=" * 60)
    print("ИНФОРМАЦИЯ О МОДЕЛИ")
    print("=" * 60)
    print(f"Эксперимент: {experiment_name}")
    print(f"Версия весов: {version}.pt")
    if hasattr(model, 'names'):
        print(f"Классы: {model.names}")
    if hasattr(model, 'overrides') and 'imgsz' in model.overrides:
        print(f"Размер изображений: {model.overrides['imgsz']}")
    
    # Пример использования или предсказание
    predict_example(model, source=args.source, device=args.device)
    
    print("\n" + "=" * 60)
    print("ГОТОВО")
    print("=" * 60)
    print(f"Модель загружена и готова к использованию")
    print(f"Используйте переменную 'model' для предсказаний")


if __name__ == "__main__":
    main()

