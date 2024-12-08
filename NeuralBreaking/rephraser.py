import torch
from torch.quantization import quantize_dynamic
from transformers import GPT2Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm
import re
import os


def get_size_of_model(model):
    total_size = 0
    for name, param in model.named_parameters():
        param_size = param.nelement() * param.element_size()
        total_size += param_size
        print(f"Параметр: {name}, Тип данных: {param.dtype}, Размер: {param.size()}, Общий байт: {param_size}")
    print(f"Общий размер модели: {total_size / (1024 * 1024):.2f} МБ")


def quantize_model(model):
    model = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    return model


def summarize_text(input_text):
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to('cpu')
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            eos_token_id=tokenizer.eos_token_id,
            num_beams=8,
            min_new_tokens=20,
            max_new_tokens=60,
            do_sample=True,
            no_repeat_ngram_size=4,
            top_p=0.9
        )
    print('Текст сокращён')
    return tokenizer.decode(outputs[0][1:])


def is_valid_filename(filename):
    pattern = r'^[^<>:"/\\|?*\x00-\x1F]+$'
    return len(filename) <= 255 and re.match(pattern, filename) is not None


def make_save_data(input_f, output_f, model, tokenizer):
    if not is_valid_filename(input_f):
        return False, f'Имя входного файла "{input_f}" невалидно.'
    if not is_valid_filename(output_f):
        return False, f'Имя выходного файла "{output_f}" невалидно.'

    with open(input_f, 'r', encoding='utf-8') as infile, open(output_f, 'w', encoding='utf-8') as outfile:
        for line in tqdm(infile):
            print(line)
            line = line.strip()
            if line:
                summary = summarize_text(line, model, tokenizer)
                outfile.write(summary + '\n')


def save_quantized_model(model, tokenizer, save_directory):
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    tokenizer.save_pretrained(save_directory)
    print(f"Токенизатор сохранён в '{save_directory}'")

    model_save_path = os.path.join(save_directory, 'quantized_model.pth')
    torch.save(model, model_save_path)
    print(f"Квантованная модель сохранена в '{model_save_path}'")


def load_quantized_model(save_directory):
    tokenizer = GPT2Tokenizer.from_pretrained(save_directory, eos_token='</s>')
    print(f"Токенизатор загружен из '{save_directory}'")

    model_save_path = os.path.join(save_directory, 'quantized_model.pth')
    model = torch.load(model_save_path, map_location='cpu')
    model.eval()
    print(f"Квантованная модель загружена из '{model_save_path}'")

    return model, tokenizer


def launch_model(text, is_quant_model=True):
    if is_quant_model:
        model, tokenizer = load_quantized_model('T5-FRED-Summarizer-Q')
    else:
        model, tokenizer = (GPT2Tokenizer.from_pretrained('RussianNLP/FRED-T5-Summarizer', eos_token='</s>'),
                            T5ForConditionalGeneration.from_pretrained('RussianNLP/FRED-T5-Summarizer').to('cpu'))

    summarized_text = summarize_text(text, model, tokenizer).replace('</s>', '')

    print(summarized_text)

    return summarized_text

model, tokenizer = load_quantized_model('T5-FRED-Summarizer-Q')


if __name__ == '__main__':
    print(summarize_text("На Кубани «десятка» снесла дорожное ограждение и перевернулась, водитель погиб. Смертельная авария произошла 3 декабря около 8:20 на 72 км федеральной трассы «Кавказ» в Тихорецком районе. По предварительным данным, водитель автомобиля «ВАЗ-2110» при движении со стороны Кропоткина в Тихорецк превысил скорость и не справился с управлением. Легковушка снесла дорожное ограждение, вылетела с дороги и перевернулась. В результате ДТП водитель «десятки» скончался на месте до приезда скорой помощи. Уточняется, что за рулем был житель Гулькевичского района, мужчина на вид 30 лет. Обстоятельства происшествия устанавливаются, сообщает отдел пропаганды БДД УГИБДД России по Краснодарскому краю."))
    # Этот код предназначен для скачивания модели и её квантования, что занимает много времени. После код бессмысленен.
    allow = False
    if allow:
        torch.cuda.empty_cache()
        model, tokenizer = (GPT2Tokenizer.from_pretrained('RussianNLP/FRED-T5-Summarizer', eos_token='</s>'),
                            T5ForConditionalGeneration.from_pretrained('RussianNLP/FRED-T5-Summarizer').to('cpu'))
        print("Исходный размер модели:")
        get_size_of_model(model)

        model = quantize_model(model)
        print("\nРазмер квантованной модели:")
        get_size_of_model(model)

        quantized_model_dir = 'T5-FRED-Summarizer-Q'
        save_quantized_model(model, tokenizer, quantized_model_dir)