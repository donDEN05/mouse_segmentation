"""
Модуль для обработки изображений
Изменяет размер изображения до 1280x960 пикселей
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Union, Tuple, Optional


def resize_image(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    target_size: Tuple[int, int] = (1280, 960),
    interpolation: int = cv2.INTER_LINEAR,
    keep_aspect_ratio: bool = True,  # По умолчанию сохраняем пропорции
    background_color: Tuple[int, int, int] = (0, 0, 0)
) -> bool:
    """
    Изменяет размер изображения до указанного размера
    
    Parameters:
    -----------
    input_path : str or Path
        Путь к входному изображению
    output_path : str or Path
        Путь для сохранения обработанного изображения
    target_size : tuple of int
        Целевой размер (ширина, высота). По умолчанию (1280, 960)
    interpolation : int
        Метод интерполяции OpenCV. По умолчанию cv2.INTER_LINEAR
        Другие варианты: cv2.INTER_CUBIC (лучшее качество, медленнее),
                         cv2.INTER_AREA (лучше для уменьшения),
                         cv2.INTER_NEAREST (быстрее, но хуже качество)
    keep_aspect_ratio : bool
        Если True (по умолчанию), сохраняет пропорции и добавляет черные полосы при необходимости
        Если False, растягивает/сжимает изображение до точного размера (может исказить пропорции)
    background_color : tuple of int
        Цвет фона для заполнения (B, G, R). Используется только если keep_aspect_ratio=True
        
    Returns:
    --------
    bool
        True если обработка успешна, False в случае ошибки
        
    Examples:
    ---------
    >>> # Изменение размера с сохранением пропорций (по умолчанию)
    >>> resize_image('input.jpg', 'output.jpg')
    
    >>> # С растягиванием до точного размера (может исказить пропорции)
    >>> resize_image('input.jpg', 'output.jpg', keep_aspect_ratio=False)
    
    >>> # С другим размером
    >>> resize_image('input.jpg', 'output.jpg', target_size=(1920, 1080))
    """
    try:
        input_path = Path(input_path)
        output_path = Path(output_path)
        
        # Проверка существования входного файла
        if not input_path.exists():
            print(f"❌ Входной файл не найден: {input_path}")
            return False
        
        # Загрузка изображения
        image = cv2.imread(str(input_path))
        if image is None:
            print(f"❌ Не удалось загрузить изображение: {input_path}")
            return False
        
        original_height, original_width = image.shape[:2]
        target_width, target_height = target_size
        
        print(f"📷 Оригинальный размер: {original_width}x{original_height}")
        print(f"🎯 Целевой размер: {target_width}x{target_height}")
        
        if keep_aspect_ratio:
            # Сохраняем пропорции
            # Вычисляем коэффициенты масштабирования
            scale_width = target_width / original_width
            scale_height = target_height / original_height
            scale = min(scale_width, scale_height)  # Используем меньший коэффициент
            
            # Вычисляем новые размеры
            new_width = int(original_width * scale)
            new_height = int(original_height * scale)
            
            # Изменяем размер изображения
            resized_image = cv2.resize(image, (new_width, new_height), interpolation=interpolation)
            
            # Создаем изображение с целевым размером и заполняем фоном
            result_image = np.full((target_height, target_width, 3), background_color, dtype=np.uint8)
            
            # Вычисляем позицию для центрирования
            y_offset = (target_height - new_height) // 2
            x_offset = (target_width - new_width) // 2
            
            # Вставляем измененное изображение в центр
            result_image[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized_image
            
            print(f"✓ Изображение изменено с сохранением пропорций: {new_width}x{new_height} → {target_width}x{target_height}")
            
        else:
            # Просто растягиваем/сжимаем до точного размера
            result_image = cv2.resize(image, (target_width, target_height), interpolation=interpolation)
            print(f"✓ Изображение изменено: {original_width}x{original_height} → {target_width}x{target_height}")
        
        # Создаем директорию для выходного файла, если её нет
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Сохранение результата
        success = cv2.imwrite(str(output_path), result_image)
        
        if success:
            print(f"✓ Изображение сохранено: {output_path}")
            return True
        else:
            print(f"❌ Не удалось сохранить изображение: {output_path}")
            return False
            
    except Exception as e:
        print(f"❌ Ошибка при обработке изображения: {e}")
        return False


def resize_image_simple(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    target_width: int = 1280,
    target_height: int = 960
) -> bool:
    """
    Упрощенная версия функции resize_image для быстрого изменения размера до 1280x960
    Масштабирует изображение с сохранением пропорций (letterboxing)
    
    Parameters:
    -----------
    input_path : str or Path
        Путь к входному изображению
    output_path : str or Path
        Путь для сохранения обработанного изображения
    target_width : int
        Целевая ширина (по умолчанию 1280)
    target_height : int
        Целевая высота (по умолчанию 960)
        
    Returns:
    --------
    bool
        True если обработка успешна, False в случае ошибки
    """
    return resize_image(
        input_path=input_path,
        output_path=output_path,
        target_size=(target_width, target_height),
        keep_aspect_ratio=True  # Сохраняем пропорции по умолчанию
    )


def resize_image_preserve_aspect(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    target_width: int = 1280,
    target_height: int = 960,
    background_color: Tuple[int, int, int] = (0, 0, 0)
) -> bool:
    """
    Изменяет размер изображения до 1280x960 с сохранением пропорций
    
    Parameters:
    -----------
    input_path : str or Path
        Путь к входному изображению
    output_path : str or Path
        Путь для сохранения обработанного изображения
    target_width : int
        Целевая ширина (по умолчанию 1280)
    target_height : int
        Целевая высота (по умолчанию 960)
    background_color : tuple of int
        Цвет фона (B, G, R) для заполнения. По умолчанию черный (0, 0, 0)
        
    Returns:
    --------
    bool
        True если обработка успешна, False в случае ошибки
    """
    return resize_image(
        input_path=input_path,
        output_path=output_path,
        target_size=(target_width, target_height),
        keep_aspect_ratio=True,
        background_color=background_color
    )


def process_image_batch(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    target_size: Tuple[int, int] = (1280, 960),
    keep_aspect_ratio: bool = False,
    extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
) -> Tuple[int, int]:
    """
    Обрабатывает все изображения в директории
    
    Parameters:
    -----------
    input_dir : str or Path
        Директория с входными изображениями
    output_dir : str or Path
        Директория для сохранения обработанных изображений
    target_size : tuple of int
        Целевой размер (ширина, высота). По умолчанию (1280, 960)
    keep_aspect_ratio : bool
        Сохранять ли пропорции изображений
    extensions : tuple of str
        Расширения файлов для обработки
        
    Returns:
    --------
    tuple of int
        (количество успешно обработанных, количество ошибок)
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    if not input_dir.exists():
        print(f"❌ Входная директория не найдена: {input_dir}")
        return (0, 0)
    
    # Создаем выходную директорию
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Находим все изображения
    image_files = []
    for ext in extensions:
        image_files.extend(input_dir.glob(f'*{ext}'))
        image_files.extend(input_dir.glob(f'*{ext.upper()}'))
    
    if not image_files:
        print(f"⚠ Изображения не найдены в директории: {input_dir}")
        return (0, 0)
    
    print(f"📁 Найдено изображений: {len(image_files)}")
    
    success_count = 0
    error_count = 0
    
    for image_file in image_files:
        output_file = output_dir / image_file.name
        
        print(f"\nОбработка: {image_file.name}")
        if resize_image(
            input_path=image_file,
            output_path=output_file,
            target_size=target_size,
            keep_aspect_ratio=keep_aspect_ratio
        ):
            success_count += 1
        else:
            error_count += 1
    
    print(f"\n{'='*60}")
    print(f"✓ Успешно обработано: {success_count}")
    if error_count > 0:
        print(f"❌ Ошибок: {error_count}")
    print(f"{'='*60}")
    
    return (success_count, error_count)


if __name__ == "__main__":
    """
    Пример использования модуля
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Обработка изображений - изменение размера до 1280x960",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python image_processing.py input.jpg output.jpg
  python image_processing.py input.jpg output.jpg --preserve-aspect
  python image_processing.py input_dir output_dir --batch
  python image_processing.py input.jpg output.jpg --width 1920 --height 1080
        """
    )
    
    parser.add_argument('input', help='Входной файл или директория')
    parser.add_argument('output', help='Выходной файл или директория')
    parser.add_argument('--width', type=int, default=1280, help='Целевая ширина (по умолчанию: 1280)')
    parser.add_argument('--height', type=int, default=960, help='Целевая высота (по умолчанию: 960)')
    parser.add_argument('--preserve-aspect', action='store_true', 
                       help='Сохранять пропорции изображения (добавляет черные полосы при необходимости)')
    parser.add_argument('--batch', action='store_true',
                       help='Обработать все изображения в директории')
    parser.add_argument('--background', type=int, nargs=3, default=[0, 0, 0],
                       metavar=('B', 'G', 'R'),
                       help='Цвет фона (B G R) для режима preserve-aspect (по умолчанию: 0 0 0 - черный)')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if args.batch:
        # Пакетная обработка
        success, errors = process_image_batch(
            input_dir=input_path,
            output_dir=output_path,
            target_size=(args.width, args.height),
            keep_aspect_ratio=args.preserve_aspect
        )
    else:
        # Обработка одного файла
        if args.preserve_aspect:
            success = resize_image_preserve_aspect(
                input_path=input_path,
                output_path=output_path,
                target_width=args.width,
                target_height=args.height,
                background_color=tuple(args.background)
            )
        else:
            success = resize_image_simple(
                input_path=input_path,
                output_path=output_path,
                target_width=args.width,
                target_height=args.height
            )
        
        if success:
            print("\n✓ Обработка завершена успешно")
        else:
            print("\n❌ Ошибка при обработке")
            exit(1)

