"""
Скрипт для дообучения модели YOLO11 на датасете epm_1
Использует предобученную модель YOLO11 и дообучает на кадрах из видео epm_1.mp4
Интегрирован с MLflow для логирования метрик, параметров и градиентов
"""
from ultralytics import YOLO
from pathlib import Path
import argparse
import mlflow
import torch
import numpy as np
import re
import os

# Попытка импорта torchvision для проверки совместимости
try:
    import torchvision
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False


def get_mlflow_backend_uri(mlruns_dir):
    """
    Формирует правильный backend URI для MLflow с учетом особенностей Windows
    
    Parameters:
    -----------
    mlruns_dir : str or Path
        Путь к директории mlruns
        
    Returns:
    --------
    str : Backend URI для MLflow
    """
    abs_path = os.path.abspath(mlruns_dir)
    
    if os.name == 'nt':  # Windows
        # На Windows используем формат file:///D:/path/to/mlruns (с заглавной буквой диска)
        # Преобразуем путь: D:\path\to\mlruns -> D:/path/to/mlruns
        uri_path = abs_path.replace('\\', '/')
        # Убеждаемся, что буква диска заглавная
        if len(uri_path) > 1 and uri_path[1] == ':':
            uri_path = uri_path[0].upper() + uri_path[1:]
        backend_uri = f"file:///{uri_path}"
    else:  # Unix/Linux/Mac
        backend_uri = f"file://{abs_path}"
    
    return backend_uri


def clean_metric_name(name):
    """
    Очищает имя метрики от недопустимых символов для MLflow
    
    MLflow разрешает только: alphanumerics, underscores (_), dashes (-), 
    periods (.), spaces ( ) и slashes (/)
    
    Parameters:
    -----------
    name : str
        Исходное имя метрики
        
    Returns:
    --------
    str : Очищенное имя метрики
    """
    # Удаляем скобки и их содержимое
    name = re.sub(r'\([^)]*\)', '', name)
    # Удаляем другие недопустимые символы (оставляем только разрешенные)
    # Разрешенные: буквы, цифры, _, -, ., пробелы, /
    name = re.sub(r'[^a-zA-Z0-9_\-\.\s/]', '_', name)
    # Заменяем множественные подчеркивания на одно
    name = re.sub(r'_+', '_', name)
    # Удаляем подчеркивания в начале и конце
    name = name.strip('_')
    return name


def compute_gradient_norm(model):
    """
    Вычисляет норму градиентов модели
    
    Parameters:
    -----------
    model : torch.nn.Module
        Модель PyTorch
        
    Returns:
    --------
    float : Норма градиентов
    """
    total_norm = 0.0
    param_count = 0
    
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            param_count += 1
    
    if param_count > 0:
        total_norm = total_norm ** (1. / 2)
    else:
        total_norm = 0.0
    
    return total_norm


def convert_to_float(value):
    """
    Преобразует значение метрики в float для логирования в MLflow
    
    Parameters:
    -----------
    value : any
        Значение метрики (может быть Tensor, numpy array, float, int и т.д.)
        
    Returns:
    --------
    float : Преобразованное значение или None, если не удалось преобразовать
    """
    try:
        if isinstance(value, torch.Tensor):
            return float(value.item())
        elif isinstance(value, (np.ndarray, np.generic)):
            return float(value.item() if value.size == 1 else value.flat[0])
        elif isinstance(value, (int, float)):
            return float(value)
        else:
            # Попытка преобразования через float()
            return float(value)
    except (ValueError, TypeError, AttributeError):
        return None


def log_to_mlflow_epoch_end(trainer):
    """
    Колбэк для логирования метрик в конце каждой эпохи обучения
    
    Parameters:
    -----------
    trainer : DetectionTrainer
        Объект тренера YOLO
    """
    try:
        # Проверка активного MLflow run
        if mlflow.active_run() is None:
            return
        
        metrics_dict = {}
        
        # Получение номера эпохи (trainer.epoch - текущая эпоха)
        epoch = getattr(trainer, 'epoch', 0)
        # Проверяем, что epoch - это число (может быть 0-indexed или 1-indexed)
        if not isinstance(epoch, (int, float)) or epoch < 0:
            epoch = 0
        
        # Получение total loss (tloss - правильный атрибут для YOLO)
        if hasattr(trainer, 'tloss'):
            tloss_value = convert_to_float(trainer.tloss)
            if tloss_value is not None:
                metrics_dict['train/total_loss'] = tloss_value
        
        # Получение компонентов loss через loss_names и loss_items
        if hasattr(trainer, 'loss_names') and hasattr(trainer, 'loss_items'):
            if isinstance(trainer.loss_names, (list, tuple)) and isinstance(trainer.loss_items, (list, tuple)):
                for loss_name, loss_value in zip(trainer.loss_names, trainer.loss_items):
                    loss_float = convert_to_float(loss_value)
                    if loss_float is not None:
                        clean_name = clean_metric_name(f'train/loss_{loss_name}')
                        metrics_dict[clean_name] = loss_float
        
        # Получение метрик обучения из словаря metrics (если доступен)
        # Обрабатываем только метрики обучения (не валидационные)
        if hasattr(trainer, 'metrics') and isinstance(trainer.metrics, dict):
            for key, value in trainer.metrics.items():
                # Пропускаем валидационные метрики (они начинаются с 'metrics/' и логируются в on_val_end)
                key_lower = key.lower()
                if 'val' not in key_lower and not key_lower.startswith('metrics/'):
                    value_float = convert_to_float(value)
                    if value_float is not None:
                        clean_key = clean_metric_name(f'train/{key}')
                        metrics_dict[clean_key] = value_float
        
        # Логирование learning rate
        if hasattr(trainer, 'optimizer') and trainer.optimizer is not None:
            try:
                if len(trainer.optimizer.param_groups) > 0:
                    current_lr = trainer.optimizer.param_groups[0].get('lr', None)
                    if current_lr is not None:
                        metrics_dict['train/learning_rate'] = float(current_lr)
            except Exception:
                pass
        
        # Логирование нормы градиентов
        if hasattr(trainer, 'model') and trainer.model is not None:
            try:
                grad_norm = compute_gradient_norm(trainer.model)
                if grad_norm is not None:
                    metrics_dict['train/gradient_norm'] = float(grad_norm)
            except Exception:
                pass
        
        # Логирование метрик в MLflow
        if metrics_dict:
            # Очистка имен метрик от недопустимых символов
            clean_metrics = {}
            for key, value in metrics_dict.items():
                clean_key = clean_metric_name(key)
                clean_metrics[clean_key] = value
            
            mlflow.log_metrics(clean_metrics, step=epoch)
            print(f"[Epoch {epoch}] Logged {len(clean_metrics)} training metrics to MLflow")
            
    except Exception as e:
        print(f"⚠ Ошибка при логировании метрик обучения в MLflow: {e}")


def log_to_mlflow_val_end(trainer):
    """
    Колбэк для логирования метрик валидации
    
    Parameters:
    -----------
    trainer : DetectionTrainer
        Объект тренера YOLO
    """
    try:
        # Проверка активного MLflow run
        if mlflow.active_run() is None:
            return
        
        val_metrics = {}
        
        # Получение номера эпохи (trainer.epoch - текущая эпоха)
        epoch = getattr(trainer, 'epoch', 0)
        # Проверяем, что epoch - это число (может быть 0-indexed или 1-indexed)
        if not isinstance(epoch, (int, float)) or epoch < 0:
            epoch = 0
        
        # Получение метрик валидации из словаря metrics
        if hasattr(trainer, 'metrics') and isinstance(trainer.metrics, dict):
            metrics = trainer.metrics
            
            # Извлечение основных метрик валидации
            metric_mappings = {
                'metrics/mAP50(B)': 'val/mAP50',
                'metrics/mAP50-95(B)': 'val/mAP50-95',
                'metrics/mAP75(B)': 'val/mAP75',
                'metrics/precision(B)': 'val/precision',
                'metrics/recall(B)': 'val/recall',
            }
            
            for metric_key, metric_name in metric_mappings.items():
                if metric_key in metrics:
                    value_float = convert_to_float(metrics[metric_key])
                    if value_float is not None:
                        val_metrics[metric_name] = value_float
            
            # Логирование всех метрик, содержащих 'metrics' в ключе
            for key, value in metrics.items():
                if 'metrics' in key.lower():
                    # Пропускаем уже обработанные метрики
                    if key not in metric_mappings:
                        value_float = convert_to_float(value)
                        if value_float is not None:
                            clean_key = clean_metric_name(key)
                            val_metrics[f'val/{clean_key}'] = value_float
        
        # Получение метрик валидации через validator, если доступен
        if hasattr(trainer, 'validator'):
            validator = trainer.validator
            
            # Попытка получить метрики из results validator'а
            if hasattr(validator, 'results_dict') and isinstance(validator.results_dict, dict):
                for key, value in validator.results_dict.items():
                    value_float = convert_to_float(value)
                    if value_float is not None:
                        clean_key = clean_metric_name(f'val/{key}')
                        val_metrics[clean_key] = value_float
        
        # Логирование метрик в MLflow
        if val_metrics:
            # Очистка имен метрик от недопустимых символов
            clean_val_metrics = {}
            for key, value in val_metrics.items():
                clean_key = clean_metric_name(key)
                clean_val_metrics[clean_key] = value
            
            mlflow.log_metrics(clean_val_metrics, step=epoch)
            print(f"[Epoch {epoch}] Logged {len(clean_val_metrics)} validation metrics to MLflow")
        else:
            print(f"[Epoch {epoch}] No validation metrics to log")
            
    except Exception as e:
        print(f"⚠ Ошибка при логировании метрик валидации в MLflow: {e}")


def train_yolo_model(
    data_yaml="epm1_dataset.yaml",
    model_name="yolo11n.pt",  # Можно выбрать: yolo11n.pt, yolo11s.pt, yolo11m.pt, yolo11l.pt, yolo11x.pt
    epochs=10,
    imgsz=1280,  # Размер изображения (YOLO автоматически обрабатывает аспектное соотношение для 1280x960)
    batch=16,
    device=0,  # GPU device (0, 1, 2, 3) или "cpu"
    project="runs/detect",
    name="epm1_finetune",
    patience=50,  # Ранняя остановка при отсутствии улучшений
    save=True,
    plots=True,
    mlflow_tracking_uri=None,  # URI для MLflow tracking server (None = локальное хранилище)
    mlflow_experiment_name="yolo11_epm1_finetune"  # Имя эксперимента в MLflow
):
    """
    Дообучение модели YOLO11 на кастомном датасете
    
    Parameters:
    -----------
    data_yaml : str
        Путь к YAML файлу с конфигурацией датасета
    model_name : str
        Имя предобученной модели YOLO11
    epochs : int
        Количество эпох обучения
    imgsz : int
        Размер изображений для обучения (ширина)
    batch : int
        Размер батча
    device : int or str
        Устройство для обучения (GPU номер или "cpu")
    project : str
        Директория для сохранения результатов
    name : str
        Имя эксперимента
    patience : int
        Количество эпох для ранней остановки
    save : bool
        Сохранять ли чекпоинты
    plots : bool
        Создавать ли графики обучения
    """
    # Проверка 1: Существование конфигурационного файла
    print("=" * 60)
    print("ШАГ 1: ПРОВЕРКА КОНФИГУРАЦИОННОГО ФАЙЛА")
    print("=" * 60)
    data_yaml_path = Path(data_yaml)
    if not data_yaml_path.exists():
        raise FileNotFoundError(
            f"❌ Конфигурационный файл {data_yaml} не найден.\n"
            f"   Убедитесь, что вы создали датасет с помощью prepare_dataset.py"
        )
    print(f"✓ Конфигурационный файл найден: {data_yaml_path.absolute()}")
    
    # Проверка 2: Валидация параметров
    print("\n" + "=" * 60)
    print("ШАГ 2: ВАЛИДАЦИЯ ПАРАМЕТРОВ")
    print("=" * 60)
    if epochs <= 0:
        raise ValueError(f"❌ Количество эпох должно быть положительным, получено: {epochs}")
    if batch <= 0:
        raise ValueError(f"❌ Размер батча должен быть положительным, получено: {batch}")
    if imgsz <= 0:
        raise ValueError(f"❌ Размер изображений должен быть положительным, получено: {imgsz}")
    print(f"✓ Параметры валидны: epochs={epochs}, batch={batch}, imgsz={imgsz}")
    
    # Проверка 3: Проверка устройства (GPU/CPU) и совместимости torchvision
    print("\n" + "=" * 60)
    print("ШАГ 3: ПРОВЕРКА УСТРОЙСТВА И СОВМЕСТИМОСТИ")
    print("=" * 60)
    
    # Проверка совместимости torchvision
    try:
        import torchvision
        torchvision_version = torchvision.__version__
        torch_version = torch.__version__
        print(f"✓ PyTorch версия: {torch_version}")
        print(f"✓ TorchVision версия: {torchvision_version}")
        
        # Проверка наличия CUDA поддержки в torchvision
        if isinstance(device, int) and torch.cuda.is_available():
            try:
                # Пробуем выполнить тестовую операцию NMS на CUDA
                test_boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0]], device=f'cuda:{device}', dtype=torch.float32)
                test_scores = torch.tensor([0.9], device=f'cuda:{device}', dtype=torch.float32)
                # Эта операция может вызвать ошибку, если CUDA NMS не поддерживается
                try:
                    _ = torchvision.ops.nms(test_boxes, test_scores, 0.5)
                    print(f"✓ TorchVision поддерживает CUDA операции")
                except RuntimeError as nms_error:
                    if "torchvision::nms" in str(nms_error) and "CUDA" in str(nms_error):
                        print(f"⚠ TorchVision не поддерживает CUDA операции NMS")
                        print(f"  Это может быть из-за несовместимости версий torch/torchvision")
                        print(f"  Автоматическое переключение на CPU")
                        device = "cpu"
                    else:
                        raise
            except Exception as test_error:
                print(f"⚠ Ошибка при проверке CUDA поддержки: {test_error}")
                if isinstance(device, int):
                    print(f"  Переключение на CPU")
                    device = "cpu"
    except ImportError:
        print(f"⚠ TorchVision не найден, продолжаем без проверки")
    
    # Проверка устройства
    try:
        if isinstance(device, int):
            if torch.cuda.is_available() and device != "cpu":
                if device >= torch.cuda.device_count():
                    raise ValueError(f"❌ GPU с индексом {device} недоступен. Доступно GPU: {torch.cuda.device_count()}")
                print(f"✓ Будет использоваться GPU {device}: {torch.cuda.get_device_name(device)}")
            else:
                if device != "cpu":
                    print(f"⚠ GPU недоступна или отключена, будет использоваться CPU")
                device = "cpu"
        elif device == "cpu":
            print(f"✓ Будет использоваться CPU")
        else:
            raise ValueError(f"❌ Неверное значение device: {device}. Используйте число (индекс GPU) или 'cpu'")
    except Exception as e:
        print(f"⚠ Ошибка при проверке устройства: {e}")
        device = "cpu"
        print(f"  Переключено на CPU")
    
    print("\n" + "=" * 60)
    print("ПАРАМЕТРЫ ОБУЧЕНИЯ")
    print("=" * 60)
    print(f"Модель: {model_name}")
    print(f"Датасет: {data_yaml}")
    print(f"Эпохи: {epochs}")
    print(f"Размер изображений: {imgsz}")
    print(f"Размер батча: {batch}")
    print(f"Устройство: {device}")
    print(f"Проект: {project}")
    print(f"Имя эксперимента: {name}")
    print("=" * 60)
    
    # Проверка 4: Настройка MLflow
    print("\n" + "=" * 60)
    print("ШАГ 4: НАСТРОЙКА MLFLOW")
    print("=" * 60)
    try:
        if mlflow_tracking_uri:
            mlflow.set_tracking_uri(mlflow_tracking_uri)
            print(f"✓ MLflow tracking URI установлен: {mlflow_tracking_uri}")
        else:
            # Локальное хранилище MLflow в директории mlruns корня проекта
            # Используем тот же путь, что и в start_mlflow_ui.py
            BASE_DIR = Path(__file__).parent.absolute()
            MLRUNS_DIR = BASE_DIR / "mlruns"
            
            # Создаем директорию, если её нет
            MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
            
            # Формируем правильный backend URI (как в start_mlflow_ui.py)
            backend_uri = get_mlflow_backend_uri(MLRUNS_DIR)
            mlflow.set_tracking_uri(backend_uri)
            print(f"✓ Локальное хранилище MLflow: {MLRUNS_DIR.absolute()}")
            print(f"✓ Backend URI: {backend_uri}")
        
        # Установка или создание эксперимента
        try:
            experiment = mlflow.get_experiment_by_name(mlflow_experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(mlflow_experiment_name)
                print(f"✓ Создан новый эксперимент MLflow: {mlflow_experiment_name}")
            else:
                experiment_id = experiment.experiment_id
                print(f"✓ Используется существующий эксперимент MLflow: {mlflow_experiment_name} (ID: {experiment_id})")
        except Exception as e:
            raise RuntimeError(f"❌ Ошибка при настройке MLflow эксперимента '{mlflow_experiment_name}': {e}")
    except Exception as e:
        raise RuntimeError(f"❌ Ошибка при инициализации MLflow: {e}")
    
    # Проверка 5: Загрузка модели
    print("\n" + "=" * 60)
    print("ШАГ 5: ЗАГРУЗКА МОДЕЛИ")
    print("=" * 60)
    try:
        print(f"Загрузка модели {model_name}...")
        model = YOLO(model_name)
        print(f"✓ Модель {model_name} успешно загружена")
    except Exception as e:
        raise RuntimeError(f"❌ Ошибка при загрузке модели {model_name}: {e}\n"
                         f"   Убедитесь, что модель доступна или будет загружена автоматически")
    
    # Параметры обучения для логирования (будут обновлены после вызова train)
    training_params = {
        'model_name': model_name,
        'epochs': epochs,
        'imgsz': imgsz,
        'batch': batch,
        'device': str(device),
        'patience': patience,
        'data_yaml': data_yaml,
        'optimizer': 'Adam',
        'lr0': 0.01,
        'weight_decay': 0.0005,
        'warmup_epochs': 5,
        'box_loss_weight': 7.5,
        'cls_loss_weight': 0.5,
        'dfl_loss_weight': 1.5,
    }
    
    # Проверка 6: Запуск обучения
    print("\n" + "=" * 60)
    print("ШАГ 6: ЗАПУСК ОБУЧЕНИЯ")
    print("=" * 60)
    
    with mlflow.start_run(experiment_id=experiment_id, run_name=name):
        # Добавление колбэков для логирования перед обучением
        try:
            model.add_callback("on_train_epoch_end", log_to_mlflow_epoch_end)
            model.add_callback("on_val_end", log_to_mlflow_val_end)
            print("✓ Колбэки MLflow успешно добавлены")
        except Exception as e:
            print(f"⚠ Предупреждение: не удалось добавить некоторые колбэки: {e}")
        
        # Запуск обучения
        print(f"\nНачало обучения...")
        try:
            results = model.train(
                data=data_yaml,
                epochs=epochs,
                imgsz=imgsz,
                batch=batch,
                device=device,
                project=project,
                name=name,
                patience=patience,
                save=save,
                plots=plots,
                verbose=True,
                # Дополнительные параметры оптимизации
                lr0=0.01,  # Начальная скорость обучения
                optimizer="Adam",  # Оптимизатор (SGD или Adam)
                weight_decay=0.0005,
                warmup_epochs=5,  # Количество эпох разогрева
                box=7.5,  # Вес потери боксов
                cls=0.5,  # Вес потери классификации
                dfl=1.5,  # Вес потери распределения фокальной длины
            )
            print("✓ Обучение успешно завершено")
        except RuntimeError as e:
            error_str = str(e)
            # Проверка на ошибку torchvision::nms с CUDA
            if ("torchvision::nms" in error_str or "torchvision::nms" in error_str.lower()) and \
               ("CUDA" in error_str or "cuda" in error_str.lower()) and \
               device != "cpu":
                print(f"\n" + "=" * 60)
                print("⚠ ОБНАРУЖЕНА ОШИБКА СОВМЕСТИМОСТИ")
                print("=" * 60)
                print(f"TorchVision не поддерживает CUDA операции NMS")
                print(f"Это часто происходит при несовместимости версий torch/torchvision")
                print(f"или когда torchvision был установлен без CUDA поддержки")
                print(f"\nПереключение на CPU и повторная попытка обучения...")
                print("=" * 60 + "\n")
                device = "cpu"
                # Обновляем параметры для логирования
                training_params['device'] = str(device)
                try:
                    results = model.train(
                        data=data_yaml,
                        epochs=epochs,
                        imgsz=imgsz,
                        batch=batch,
                        device=device,
                        project=project,
                        name=name,
                        patience=patience,
                        save=save,
                        plots=plots,
                        verbose=True,
                        lr0=0.01,
                        optimizer="Adam",
                        weight_decay=0.0005,
                        warmup_epochs=5,
                        box=7.5,
                        cls=0.5,
                        dfl=1.5,
                    )
                    print("\n" + "=" * 60)
                    print("✓ ОБУЧЕНИЕ УСПЕШНО ЗАВЕРШЕНО НА CPU")
                    print("=" * 60)
                    print("\nРекомендация: Для ускорения обучения установите совместимые версии:")
                    print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
                    print("=" * 60 + "\n")
                except Exception as retry_error:
                    raise RuntimeError(
                        f"❌ Ошибка при повторной попытке обучения на CPU: {retry_error}\n"
                        f"   Оригинальная ошибка CUDA: {error_str[:200]}..."
                    )
            else:
                raise RuntimeError(f"❌ Ошибка во время обучения: {e}")
        except Exception as e:
            error_str = str(e)
            # Дополнительная проверка для любых других типов исключений
            if ("torchvision::nms" in error_str or "torchvision::nms" in error_str.lower()) and \
               ("CUDA" in error_str or "cuda" in error_str.lower()) and \
               device != "cpu":
                print(f"\n⚠ Обнаружена ошибка совместимости CUDA с torchvision::nms")
                print(f"  Переключение на CPU...")
                device = "cpu"
                training_params['device'] = str(device)
                # Можно попробовать еще раз, но лучше просто сообщить об ошибке
                raise RuntimeError(
                    f"❌ Ошибка совместимости torchvision::nms с CUDA.\n"
                    f"   Переключитесь на CPU вручную: --device cpu\n"
                    f"   Или установите совместимые версии torch/torchvision с CUDA поддержкой.\n"
                    f"   Оригинальная ошибка: {error_str[:300]}..."
                )
            raise RuntimeError(f"❌ Ошибка во время обучения: {e}")
        
        # Обновление параметров обучения после train для точного логирования
        try:
            if hasattr(model, 'trainer') and hasattr(model.trainer, 'args'):
                trainer_args = model.trainer.args
                training_params.update({
                    'optimizer': getattr(trainer_args, 'optimizer', 'Adam'),
                    'lr0': getattr(trainer_args, 'lr0', 0.01),
                    'weight_decay': getattr(trainer_args, 'weight_decay', 0.0005),
                    'warmup_epochs': getattr(trainer_args, 'warmup_epochs', 5),
                    'box': getattr(trainer_args, 'box', 7.5),
                    'cls': getattr(trainer_args, 'cls', 0.5),
                    'dfl': getattr(trainer_args, 'dfl', 1.5),
                })
                print("✓ Параметры обучения обновлены")
        except Exception as e:
            print(f"⚠ Предупреждение: не удалось обновить параметры обучения: {e}")
        
        # Логирование параметров обучения
        try:
            mlflow.log_params(training_params)
            print(f"✓ Параметры обучения залогированы в MLflow")
        except Exception as e:
            print(f"⚠ Предупреждение: не удалось залогировать параметры в MLflow: {e}")
    
        # Проверка 7: Валидация модели
        print("\n" + "=" * 60)
        print("ШАГ 7: ВАЛИДАЦИЯ МОДЕЛИ")
        print("=" * 60)
        try:
            print("Запуск валидации...")
            metrics = model.val()
            print("✓ Валидация успешно завершена")
            
            print(f"\nМетрики валидации:")
            print(f"  mAP50-95: {metrics.box.map:.4f}")
            print(f"  mAP50: {metrics.box.map50:.4f}")
            print(f"  mAP75: {metrics.box.map75:.4f}")
            print(f"  Precision: {metrics.box.mp:.4f}")
            print(f"  Recall: {metrics.box.mr:.4f}")
        except Exception as e:
            raise RuntimeError(f"❌ Ошибка при валидации модели: {e}")
        
        # Проверка 8: Сохранение и логирование результатов
        print("\n" + "=" * 60)
        print("ШАГ 8: СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
        print("=" * 60)
        
        best_model_path = Path(project) / name / "weights" / "best.pt"
        if best_model_path.exists():
            print(f"✓ Лучшая модель найдена: {best_model_path.absolute()}")
        else:
            # Проверяем альтернативный путь
            last_model_path = Path(project) / name / "weights" / "last.pt"
            if last_model_path.exists():
                print(f"⚠ Лучшая модель не найдена, используется последняя: {last_model_path.absolute()}")
                best_model_path = last_model_path
            else:
                print(f"⚠ Модель не найдена в ожидаемом месте")
        
        # Логирование финальных метрик и артефактов в MLflow
        try:
            final_metrics_raw = {
                'final/mAP50-95': metrics.box.map,
                'final/mAP50': metrics.box.map50,
                'final/mAP75': metrics.box.map75,
                'final/precision': metrics.box.mp,
                'final/recall': metrics.box.mr,
            }
            # Преобразование метрик в float и очистка имен
            final_metrics = {}
            for key, value in final_metrics_raw.items():
                value_float = convert_to_float(value)
                if value_float is not None:
                    clean_key = clean_metric_name(key)
                    final_metrics[clean_key] = value_float
            
            if final_metrics:
                mlflow.log_metrics(final_metrics)
                print("✓ Финальные метрики залогированы в MLflow")
            else:
                print("⚠ Не удалось преобразовать финальные метрики")
            
            if best_model_path.exists():
                mlflow.log_artifact(str(best_model_path), artifact_path="model")
                print(f"✓ Модель залогирована как артефакт в MLflow")
            
            mlflow.log_param('best_model_path', str(best_model_path))
            
            mlflow_uri = mlflow.get_tracking_uri()
            run_id = mlflow.active_run().info.run_id
            print(f"\n✓ MLflow Run ID: {run_id}")
            print(f"  Tracking URI: {mlflow_uri}")
            if mlflow_tracking_uri and 'http' in mlflow_tracking_uri:
                print(f"  Просмотр результатов: {mlflow_tracking_uri}/#/experiments/{experiment_id}/runs/{run_id}")
        except Exception as e:
            print(f"⚠ Предупреждение: не удалось залогировать финальные результаты в MLflow: {e}")
        
        print("\n" + "=" * 60)
        print("ОБУЧЕНИЕ УСПЕШНО ЗАВЕРШЕНО")
        print("=" * 60)
    
    return model, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Дообучение YOLO11 на датасете epm_1")
    parser.add_argument(
        "--data",
        type=str,
        default="epm1_dataset.yaml",
        help="Путь к YAML файлу с конфигурацией датасета"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11n.pt",
        choices=["yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt"],
        help="Предобученная модель YOLO11"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Количество эпох обучения"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=1280,
        help="Размер изображений (используется для обеих сторон, аспектное соотношение сохраняется)"
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Размер батча"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Устройство для обучения (GPU номер или 'cpu')"
    )
    parser.add_argument(
        "--name",
        type=str,
        default="epm1_finetune",
        help="Имя эксперимента"
    )
    parser.add_argument(
        "--mlflow_uri",
        type=str,
        default=None,
        help="URI для MLflow tracking server (по умолчанию локальное хранилище)"
    )
    parser.add_argument(
        "--mlflow_experiment",
        type=str,
        default="yolo11_epm1_finetune",
        help="Имя эксперимента в MLflow"
    )
    
    args = parser.parse_args()
    
    # Преобразование device в int или оставление строки "cpu"
    device = int(args.device) if args.device.isdigit() else args.device
    
    # Запуск обучения
    train_yolo_model(
        data_yaml=args.data,
        model_name=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        name=args.name,
        mlflow_tracking_uri=args.mlflow_uri,
        mlflow_experiment_name=args.mlflow_experiment
    )

