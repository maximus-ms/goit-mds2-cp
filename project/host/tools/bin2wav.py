#!/usr/bin/env python3
"""
Конвертер bin файлу в WAV формат.
Bin файл містить інтерлівовані samples з кількох каналів.
Підтримує 16-bit та 32-bit samples.
"""

import struct
import sys
import argparse
import os
import zlib
import ctypes


def calculate_crc32(data):
    """
    Розраховує CRC32 контрольну суму (як в ESP-IDF esp_rom_crc32_le).
    Використовує zlib.crc32 з початковим значенням 0xFFFFFFFF.
    """
    crc = zlib.crc32(data, 0xFFFFFFFF) & 0xFFFFFFFF
    return crc


def calculate_simple_sum(data):
    """
    Розраховує просту суму всіх байтів.
    """
    return sum(data)

def sample_processor(sample, sample_width):
    """Обробляє семпл."""
    if sample_width == 16:
        return ctypes.c_int16(sample).value
    else:
        return ctypes.c_int32(sample).value


def samples_normalizer(samples, sample_width_bits):
    """
    Нормалізує семпли до відповідного діапазону залежно від sample_width_bits.
    
    Args:
        samples: Список семплів для нормалізації
        sample_width_bits: Ширина семплу в бітах (16 або 32)
    """
    max_sample = max(samples)
    min_sample = min(samples)
    max_abs = max(abs(max_sample), abs(min_sample))
    
    # Визначаємо діапазон для нормалізації залежно від sample_width_bits
    if sample_width_bits == 16:
        # 16-bit signed integer: [-32768, 32767]
        max_range = 32767
        min_range = -32768
    elif sample_width_bits == 32:
        # 32-bit signed integer: [-2147483647, 2147483647]
        max_range = 2147483647
        min_range = -2147483647
    else:
        # Для інших розмірів використовуємо максимальний діапазон
        max_range = 2147483647
        min_range = -2147483647
    
    if max_abs > 0:
        scale_factor = float(max_range) / max_abs
        normalized_samples = []
        cur_type = ctypes.c_int16 if sample_width_bits == 16 else ctypes.c_int32
        for sample in samples:
            normalized = int(sample * scale_factor)
            normalized = max(min_range, min(max_range, normalized))
            normalized_samples.append(cur_type(normalized).value)
        samples = normalized_samples

    return samples


def bin_to_wav(input_file, output_file, num_channels=2, sample_rate=44100, sample_width=32,
                expected_crc32=0):
    """
    Конвертує bin файл в WAV формат.
    
    Args:
        input_file: Шлях до вхідного bin файлу
        output_file: Шлях до вихідного WAV файлу
        num_channels: Кількість каналів (1=mono, 2=stereo, тощо)
        sample_rate: Частота семплювання (Гц)
        sample_width: Ширина семплу в бітах (16 або 32)
        expected_crc32: Очікувана CRC32 контрольна сума (hex рядок або int)
    """
    # Перевірка існування файлу
    if not os.path.exists(input_file):
        print(f"Помилка: файл '{input_file}' не знайдено")
        return False
    
    # Читаємо bin файл
    with open(input_file, 'rb') as f:
        data = f.read()
    
    file_size = len(data)
    
    calculated_crc32 = calculate_crc32(data)
    calculated_sum = calculate_simple_sum(data)
    
    # Перевірка контрольної суми
    checksum_valid = True
    if expected_crc32 != 0:
        # Конвертуємо hex рядок в int якщо потрібно
        if isinstance(expected_crc32, str):
            # Видаляємо префікс 0x якщо є
            expected_crc32 = expected_crc32.replace('0x', '').replace('0X', '')
            try:
                expected_crc32 = int(expected_crc32, 16)
            except ValueError:
                print(f"Помилка: некоректний формат CRC32: {expected_crc32}")
                return False
        elif isinstance(expected_crc32, int):
            pass
        else:
            print(f"Помилка: некоректний тип CRC32")
            return False
        
        if calculated_crc32 != expected_crc32:
            print(f"CRC32 mismatch: expected 0x{expected_crc32:08x}, got 0x{calculated_crc32:08x}")
            checksum_valid = False
            response = input("Continue conversion? (y/n): ")
            if response.lower() != 'y':
                return False
    
    # Конвертуємо біти в байти для розрахунків
    sample_width_bytes = sample_width // 8
    
    # Перевірка розміру
    sample_size = sample_width_bytes * num_channels  # Розмір одного "кадру" (frame) в байтах
    if file_size % sample_size != 0:
        print(f"Попередження: розмір файлу ({file_size}) не кратний розміру кадру ({sample_size})")
        # Обрізаємо до кратного розміру
        file_size = (file_size // sample_size) * sample_size
        data = data[:file_size]
    
    num_samples = file_size // sample_size
    
    # Розпаковуємо дані залежно від sample_width (в бітах)
    samples = []
    unpack_format = {
        16: '<h',  # 16-bit signed
        32: '<i'   # 32-bit signed
    }
    
    if sample_width not in unpack_format:
        print(f"Помилка: непідтримуваний розмір семплу: {sample_width} біт")
        return False
    
    fmt = unpack_format[sample_width]
    for i in range(0, file_size, sample_width_bytes):
        # Читаємо signed integer відповідного розміру (little-endian)
        sample = struct.unpack(fmt, data[i:i+sample_width_bytes])[0]
        samples.append(sample_processor(sample, sample_width))
    
    samples = samples_normalizer(samples, sample_width)
    
    # WAV підтримує 32-bit float або 16-bit/24-bit/32-bit integer
    # Використовуємо відповідний формат залежно від sample_width
    
    # Створюємо WAV файл
    with open(output_file, 'wb') as wav_file:
        # WAV заголовок
        # RIFF chunk
        wav_file.write(b'RIFF')
        # Розмір файлу - 8 (буде заповнено пізніше)
        wav_file.write(struct.pack('<I', 0))
        wav_file.write(b'WAVE')
        
        # fmt chunk
        wav_file.write(b'fmt ')
        fmt_chunk_size = 16  # Розмір fmt chunk для PCM
        wav_file.write(struct.pack('<I', fmt_chunk_size))
        audio_format = 1  # 1 = PCM
        wav_file.write(struct.pack('<H', audio_format))
        wav_file.write(struct.pack('<H', num_channels))
        wav_file.write(struct.pack('<I', sample_rate))
        byte_rate = sample_rate * num_channels * sample_width_bytes
        wav_file.write(struct.pack('<I', byte_rate))
        block_align = num_channels * sample_width_bytes
        wav_file.write(struct.pack('<H', block_align))
        bits_per_sample = sample_width  # sample_width вже в бітах
        wav_file.write(struct.pack('<H', bits_per_sample))
        
        # data chunk
        wav_file.write(b'data')
        data_size = len(samples) * sample_width_bytes
        wav_file.write(struct.pack('<I', data_size))
        
        # Записуємо дані залежно від sample_width (в бітах)
        pack_format = {
            16: '<h',  # 16-bit signed
            32: '<i'   # 32-bit signed
        }
        
        fmt = pack_format[sample_width]
        for sample in samples:
            # Записуємо signed integer відповідного розміру (little-endian)
            wav_file.write(struct.pack(fmt, sample))
        
        # Оновлюємо розмір RIFF chunk
        file_size_total = wav_file.tell() - 8
        wav_file.seek(4)
        wav_file.write(struct.pack('<I', file_size_total))
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Конвертує bin файл з інтерлівованими samples в WAV формат',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Приклади використання:
  %(prog)s input.bin output.wav
  %(prog)s input.bin output.wav --channels 2 --rate 44100
  %(prog)s input.bin output.wav -c 1 -r 8000 -w 16
  %(prog)s input.bin output.wav --crc32 0x12345678
  %(prog)s input.bin output.wav -C 12345678 -w 16
        """
    )
    
    parser.add_argument('input', help='Вхідний bin файл')
    parser.add_argument('output', help='Вихідний WAV файл')
    parser.add_argument('-c', '--channels', type=int, default=2,
                        help='Кількість каналів (за замовчуванням: 2)')
    parser.add_argument('-r', '--rate', type=int, default=32000,
                        help='Частота семплювання в Гц (за замовчуванням: 44100)')
    parser.add_argument('-w', '--width', type=int, default=32,
                        help='Ширина семплу в бітах (16 або 32, за замовчуванням: 32)')
    parser.add_argument('-C', '--crc32', type=str, default=0,
                        help='Очікувана CRC32 контрольна сума (hex, наприклад: 0x12345678 або 12345678)')

    
    args = parser.parse_args()
    
    # Перевірка параметрів
    if args.channels < 1:
        print("Помилка: кількість каналів повинна бути >= 1")
        sys.exit(1)
    
    if args.crc32 != 0:
        args.crc32 = eval(args.crc32)

    if args.rate < 1:
        print("Помилка: частота семплювання повинна бути >= 1")
        sys.exit(1)
    
    if args.width not in [16, 32]:
        print("Помилка: ширина семплу повинна бути 16 або 32 біти")
        sys.exit(1)
    
    success = bin_to_wav(
        args.input,
        args.output,
        num_channels=args.channels,
        sample_rate=args.rate,
        sample_width=args.width,
        expected_crc32=args.crc32,
    )
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
