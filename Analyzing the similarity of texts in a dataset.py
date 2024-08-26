import csv
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import time
import os
import h5py

def all_divisors(n):
    """
    Функция находит все делители числа n.
    
    Связанные функции: нет.
    
    Выполняет:
    - Находит все делители заданного числа n.
    
    Логика работы:
    - Используется цикл для проверки делителей от 1 до квадратного корня числа n.
    - Добавляет найденные делители в множество, чтобы избежать дублирования.
    
    Пример использования:
    >>> all_divisors(28)
    [1, 2, 4, 7, 14, 28]
    """
    divisors = set()  # Создаем множество для хранения делителей
    for i in range(1, int(n**0.5) + 1):  # Итерация от 1 до квадратного корня числа n
        if n % i == 0:  # Если i является делителем n
            divisors.add(i)  # Добавляем делитель i
            divisors.add(n // i)  # Добавляем сопряженный делитель n // i
    return sorted(divisors)  # Преобразуем множество в отсортированный список


def initialize_model():
    """
    Функция инициализирует модель SentenceTransformer.
    
    Связанные функции: encode_text().
    
    Выполняет:
    - Загружает и возвращает модель SentenceTransformer.
    
    Логика работы:
    - Инициализирует модель с использованием заданного предобученного модуля.
    
    Пример использования:
    >>> model = initialize_model()
    >>> model.encode("example text")
    tensor([[-0.0304, -0.0057, ...]])
    """
    start_time = time.time()  # Засекаем время начала инициализации модели
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Инициализируем модель
    end_time = time.time()  # Засекаем время окончания инициализации модели
    print(f"Время активации модели кодировки: {end_time - start_time} секунд, произведено успешно")
    return model  # Возвращаем инициализированную модель


def encode_text(model, file_path, batch_size, output_dir, max_rows):
    """
    Функция кодирует текст в эмбеддинги и сохраняет их в HDF5 файлы.
    
    Связанные функции: initialize_model(), calculate_cosine_similarity().
    
    Выполняет:
    - Кодирует текст из CSV файла в эмбеддинги с использованием модели SentenceTransformer и сохраняет их в отдельные файлы.
    
    Аргументы:
    - model: Объект модели SentenceTransformer.
    - file_path: Путь к CSV файлу с текстами для кодирования.
    - batch_size: Размер пакета для обработки.
    - output_dir: Путь к директории для сохранения файлов HDF5.
    
    Пример использования:
    >>> encode_text(model, 'data.csv', 100, 'embeddings/')
    """
    start_time = time.time()  # Засекаем время начала кодирования
    
    loader = DataLoader(file_path, batch_size, max_rows)  # Инициализация загрузчика данных
    num_batches = loader.get_batches_count()  # Количество партий
    
    total_texts = loader.get_total_count()  # Общее количество текстов
    
    for i, batch in enumerate(loader):  # Обрабатываем текст партиями
        start_timeIterate = time.time()  # Засекаем время начала обработки очередной партии
        embeddings = model.encode(batch, convert_to_tensor=True)  # Кодируем партию
        
        # Определяем имя файла для сохранения эмбеддингов
        output_file = f"{output_dir}/embeddings_batch_{i}.h5"
        # Сохраняем эмбеддинги в HDF5 файл
        with h5py.File(output_file, 'w') as f:
            f.create_dataset('embeddings', data=embeddings.cpu().numpy())
        
        end_timeIterate = time.time()  # Засекаем время окончания обработки партии
        
        # Количество обработанных текстов
        processed_texts = min((i + 1) * batch_size, total_texts)
        
        # Время кодирования
        time_taken = end_timeIterate - start_timeIterate
        
        # Индикатор времени кодирования партии
        print(f"Время кодирования для партии {i + 1}/{max_rows/batch_size}: {time_taken:.2f} секунд.")
        print(f"Обработано {i*batch_size + batch_size} текстов из {max_rows}. Продолжаю работу...")
    
    end_time = time.time()  # Засекаем время окончания кодирования
    print(f"Время кодирования текста: {end_time - start_time:.2f} секунд, произведено успешно.")   
 

def calculate_cosine_similarity(embeddings_dir, batch_size, output_directory):
    """
    Функция вычисляет косинусное сходство между эмбеддингами, хранящимися в отдельных HDF5 файлах, и сохраняет результаты.
    
    Выполняет:
    - Для каждого файла эмбеддингов из указанной директории вычисляет косинусное сходство с эмбеддингами из всех файлов,
      включая сам себя.
    - Результаты сохраняются в отдельных HDF5-файлах в указанной директории.
    
    Аргументы:
    - embeddings_dir: Директория, содержащая файлы эмбеддингов.
    - batch_size: Размер пакета для обработки.
    - output_directory: Директория для сохранения файлов с косинусным сходством.
    
    Пример использования:
    >>> calculate_cosine_similarity('D:\\temp\\embeddings\\', 100, 'D:\\temp\\cosine_scores\\')
    """
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    # Получаем список всех файлов эмбеддингов в директории
    files = sorted([f for f in os.listdir(embeddings_dir) if f.endswith('.h5')])
    num_files = len(files)
    
    start_time = time.time()  # Засекаем время начала вычисления
    
    for idx, file_name in enumerate(files):
        file_path = os.path.join(embeddings_dir, file_name)
        
        # Засекаем время загрузки эмбеддингов
        load_start_time = time.time()
        with h5py.File(file_path, 'r') as h5_file:
            batch_embeddings = np.array(h5_file['embeddings'])  # Загружаем текущие эмбеддинги
        load_end_time = time.time()
        print(f"Файл {file_name} загружен успешно. Время загрузки: {load_end_time - load_start_time:.2f} секунд.")
        
        cosine_scores_all = []
        
        # Обрабатываем текущие эмбеддинги со всеми остальными файлами
        for other_idx, other_file_name in enumerate(files):
            other_file_path = os.path.join(embeddings_dir, other_file_name)
            
            # Засекаем время загрузки других эмбеддингов
            other_load_start_time = time.time()
            with h5py.File(other_file_path, 'r') as h5_other_file:
                other_embeddings = np.array(h5_other_file['embeddings'])  # Загружаем эмбеддинги из другого файла
            other_load_end_time = time.time()
            print(f"Другой файл {other_file_name} загружен успешно. Время загрузки: {other_load_end_time - other_load_start_time:.2f} секунд.")
            
            batch_scores = util.pytorch_cos_sim(batch_embeddings, other_embeddings)
            cosine_scores_all.append(batch_scores.cpu().numpy())
        
        # Сохраняем результаты для текущего файла
        output_file = os.path.join(output_directory, f'cosine_scores_{idx}.h5')
        with h5py.File(output_file, 'w') as h5_output_file:
            # Сохраняем каждый набор данных в виде отдельного набора данных HDF5
            for i, scores in enumerate(cosine_scores_all):
                h5_output_file.create_dataset(f'cosine_scores_{i}', data=scores)
        
        end_timeIterate = time.time()  # Засекаем время окончания обработки текущего файла
        processed_files = idx + 1
        remaining_files = num_files - processed_files
        print(f"Время обработки для файла {file_name}: {end_timeIterate - start_time:.2f} секунд.")
        print(f"Обработано {processed_files} файлов, осталось {remaining_files} файлов.")
        print(f"Результаты сохранены в {output_file}.")
    
    end_time = time.time()  # Засекаем время окончания вычисления
    print(f"Общее время вычисления: {end_time - start_time:.2f} секунд.")


def cleanup_temp_files(output_directory, num_batches):
    """
    Функция удаляет временные файлы с результатами косинусного сходства.
    
    Связанные функции: calculate_cosine_similarity().
    
    Выполняет:
    - Удаляет временные файлы, содержащие промежуточные результаты.
    
    Логика работы:
    - Использует цикл для удаления всех временных файлов.
    
    Пример использования:
    >>> cleanup_temp_files('D:\\temp\\cosine_scores\\', 10)
    """
    for i in range(num_batches):  # Удаляем все временные файлы
        os.remove(f'{output_directory}cosine_scores_batch_{i}.npy')

def save_cosine_similarity_to_csv(cosine_scores, file_path):
    """
    Функция сохраняет матрицу косинусного сходства в CSV файл.
    
    Связанные функции: main().
    
    Выполняет:
    - Записывает матрицу косинусного сходства в CSV файл.
    
    Логика работы:
    - Использует csv.writer для записи данных в файл.
    
    Пример использования:
    >>> save_cosine_similarity_to_csv(cosine_scores, 'D:\\cosine_similarity_matrix.csv')
    """
    start_time = time.time()  # Засекаем время начала записи
    with open(file_path, 'w', newline='') as csvfile:  # Открываем CSV файл для записи
        writer = csv.writer(csvfile)  # Создаем объект writer для записи
        for row in cosine_scores:  # Записываем каждую строку матрицы в файл
            writer.writerow(row)
    end_time = time.time()  # Засекаем время окончания записи
    print(f"Время записи матрицы сходства в CSV: {end_time - start_time} секунд, произведено успешно")

def plot_histogram(directory_path):
    """
    Функция создает и отображает гистограмму косинусного сходства из файлов в указанной директории.
    
    Выполняет:
    - Открывает и обрабатывает все HDF5-файлы в указанной директории.
    - Извлекает верхний треугольник матриц косинусного сходства и обновляет гистограмму.
    
    Аргументы:
    - directory_path: Путь к директории, содержащей HDF5-файлы с результатами косинусного сходства.
    
    Пример использования:
    >>> plot_histogram('D:\\temp\\cosine_scores')
    """
    start_time = time.time()  # Засекаем время начала обработки
    
    # Инициализируем переменные для гистограммы
    bin_edges = np.linspace(0, 1, 101)  # Определяем границы бинов
    hist_data = np.zeros(len(bin_edges) - 1)  # Инициализируем массив для данных гистограммы
    
    # Обрабатываем все HDF5-файлы в указанной директории
    for filename in os.listdir(directory_path):
        if filename.endswith('.h5'):
            file_path = os.path.join(directory_path, filename)
            print(f"Обрабатывается файл: {file_path}")
            
            with h5py.File(file_path, 'r') as h5_file:
                for dataset_name in h5_file:
                    cosine_scores = np.array(h5_file[dataset_name])
                    upper_triangle = np.triu(cosine_scores, k=1)  # Извлекаем верхний треугольник матрицы
                    flattened_scores = upper_triangle.flatten()  # Преобразуем в плоский массив
                    filtered_scores = flattened_scores[flattened_scores != 0.0]  # Удаляем нулевые значения
                    
                    # Обновляем данные гистограммы
                    hist_data_chunk, _ = np.histogram(filtered_scores, bins=bin_edges)
                    hist_data += hist_data_chunk
    
    end_time = time.time()  # Засекаем время окончания создания гистограммы
    print(f"Время обработки и создания гистограммы: {end_time - start_time:.2f} секунд")
    
    # Создаем и отображаем гистограмму
    plt.bar(bin_edges[:-1], hist_data, width=np.diff(bin_edges), alpha=0.7, color='purple', edgecolor='black')
    plt.title('Гистограмма косинусного сходства')  # Добавляем заголовок
    plt.xlabel('Косинусное сходство')  # Добавляем метку оси X
    plt.ylabel('Количество сходств')  # Добавляем метку оси Y
    plt.axvline(x=0.5, color='red', linestyle='dashed', linewidth=1)  # Добавляем пороговое значение
    plt.grid(axis='y', alpha=0.75)  # Добавляем сетку
    plt.show()  # Отображаем гистограмму

    # Очистка памяти
    del hist_data


def check_and_create_paths(file_path, output_directory_Encode, output_directory_Cosine, cosine_csv_path):
    """
    Функция проверяет наличие файлов и директорий, и в случае их отсутствия создает их.

    Аргументы:
    - file_path: Путь к CSV файлу с текстами для кодирования.
    - output_directory_Encode: Путь к директории для сохранения файлов HDF5 и промежуточных результатов косинусного сходства.
    - output_directory_Cosine: Путь к директории для сохранения промежуточных результатов косинусного сходства.
    - cosine_csv_path: Путь к файлу CSV для сохранения матрицы косинусного сходства.

    Выполняет:
    - Проверяет и создает указанные директории.
    - Проверяет наличие файлов и уведомляет, если они отсутствуют.
    """
    # Проверка и создание директории для кодирования
    if not os.path.exists(output_directory_Encode):
        os.makedirs(output_directory_Encode)
        print(f"Создана директория: {output_directory_Encode}")
    
    # Проверка и создание директории для косинусного сходства
    if not os.path.exists(output_directory_Cosine):
        os.makedirs(output_directory_Cosine)
        print(f"Создана директория: {output_directory_Cosine}")

    # Проверка и создание файла для косинусного сходства
    if not os.path.exists(cosine_csv_path):
        with open(cosine_csv_path, 'w') as file:
            pass  # Создаем пустой файл
        print(f"Создан файл: {cosine_csv_path}")
    else:
        print(f"Файл уже существует: {cosine_csv_path}")

    # Проверка наличия исходного файла
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Исходный файл не найден: {file_path}")
    else:
        print(f"Исходный файл найден: {file_path}")


class DataLoader:
    def __init__(self, file_path, batch_size, max_rows):
        """
        Инициализация загрузчика данных.

        Аргументы:
        - file_path: Путь к CSV файлу.
        - batch_size: Размер партии данных.
        - max_rows: Максимальное количество строк для загрузки. Если None, загружаются все строки.
        """
        self.file_path = file_path
        self.batch_size = batch_size
        self.max_rows = max_rows
        self.current_chunk = 0
        self.total_chunks = None
        self.data_iterator = pd.read_csv(file_path, header=None, usecols=[0], 
                                        chunksize=batch_size, nrows=max_rows)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        """
        Загружает следующий кусок данных из файла.
        
        Возвращает:
        - Массив текстов для текущего куска данных.
        """
        try:
            chunk = next(self.data_iterator)
            self.current_chunk += 1
            return chunk[0].to_numpy()
        except StopIteration:
            raise StopIteration("Нет больше данных для загрузки.")
    
    def get_total_count(self):
        """
        Получает общее количество строк в файле.

        Возвращает:
        - Общее количество строк в файле.
        """
        if self.total_chunks is None:
            if self.max_rows is None:
                # Считаем количество строк
                self.total_chunks = sum(1 for _ in pd.read_csv(self.file_path, header=None, usecols=[0]))
            else:
                # Считаем количество строк с учетом ограничения
                self.total_chunks = min(self.max_rows, sum(1 for _ in pd.read_csv(self.file_path, header=None, usecols=[0])))
        return self.total_chunks
    
    def get_batches_count(self):
        """
        Получает количество партий в файле.

        Возвращает:
        - Количество партий в файле.
        """
        return int(np.ceil(self.get_total_count() / self.batch_size))

def emulate_encode_text(file_path, batch_size, output_dir, max_rows):
    """
    Эмулирует функцию encode_text, создавая HDF5 файлы с фиктивными эмбеддингами.
    
    Аргументы:
    - file_path: Путь к CSV файлу (не используется, но нужен для совместимости).
    - batch_size: Размер пакета для обработки.
    - output_dir: Путь к директории для сохранения файлов HDF5.
    - max_rows: Общее количество строк в CSV файле (имитируемое значение).
    
    Пример использования:
    >>> emulate_encode_text('data.csv', 100, 'embeddings/', 1000)
    """
    start_time = time.time()  # Засекаем время начала создания файлов
    
    # Создаем директорию, если она не существует
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    num_batches = (max_rows + batch_size - 1) // batch_size  # Количество партий
    
    for i in range(num_batches):
        start_timeIterate = time.time()  # Засекаем время начала обработки очередной партии
        
        # Генерируем случайные эмбеддинги для партии
        batch_embeddings = np.random.rand(batch_size, 768)  # Допустим, размерность эмбеддинга 768
        
        # Определяем имя файла для сохранения эмбеддингов
        output_file = os.path.join(output_dir, f'embeddings_batch_{i}.h5')
        
        # Сохраняем эмбеддинги в HDF5 файл
        with h5py.File(output_file, 'w') as f:
            f.create_dataset('embeddings', data=batch_embeddings)
        
        end_timeIterate = time.time()  # Засекаем время окончания обработки партии
        
        # Количество обработанных текстов
        processed_texts = min((i + 1) * batch_size, max_rows)
        
        # Время создания файла
        time_taken = end_timeIterate - start_timeIterate
        
        # Индикатор времени создания партии
        print(f"Время создания для партии {i + 1}/{num_batches}: {time_taken:.2f} секунд.")
        print(f"Обработано {processed_texts} текстов из {max_rows}. Продолжаю работу...")
    
    end_time = time.time()  # Засекаем время окончания создания файлов
    print(f"Время создания файлов: {end_time - start_time:.2f} секунд. Создание успешно завершено.")

def main():
    # Параметры
    file_path = 'D:\\LLM\\sud_resh_txt_sample_100k.csv' # Путь к файлу с датасетом
    max_rows = 100000 # Можно ставить сколько хочешь, но может быть долго и требовать безумное количество памяти на диске. Реально безумное
    batch_size_model = 1000 # Потом при вычислении сходства нужно будет запихать это в ОЗУ, осторожнее
    batch_size_cos = 10000 # Память кушает неадекватно, так что ставьте в соответсвии с доступным озу
    output_directory_Encode = 'D:\\LLM\\Encode\\'
    output_directory_Cosine = 'D:\\LLM\\Cosine_Scores\\'
    cosine_csv_path = 'D:\\LLM\\cosine_similarity_matrix.csv'

    check_and_create_paths(file_path, output_directory_Encode, output_directory_Cosine, cosine_csv_path) # Проверяет наличие директорий

    model = initialize_model() # Модель выбирается в теле функции

    # Кодирование текста и сохранение эмбеддингов
    # encode_text(model, file_path, batch_size_model, output_directory_Encode, max_rows) # Кодирование 
    emulate_encode_text('data.csv', batch_size_model, 'D:\\LLM\\Encode\\Test_embeddings', max_rows) # Создание тестового набора файлов для проверки работоспособности функций ниже


    # Вычисление косинусного сходства и сохранение результатов
    cosine_scores = calculate_cosine_similarity('D:\\LLM\\Encode\\Test_embeddings', batch_size_cos, output_directory_Cosine)

    # Сохранение матрицы косинусного сходства в CSV файл
    # save_cosine_similarity_to_csv(cosine_scores, cosine_csv_path)

    # Отображение гистограммы косинусного сходства
    plot_histogram(output_directory_Cosine)

if __name__ == "__main__":
    main()
