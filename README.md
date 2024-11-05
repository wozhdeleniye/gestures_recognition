# SHIFT train project

## Окружение(linux)
```bash
git clone https://github.com/Dragon181/SHIFT-intensive.git
cd SHIFT-intensive
python3 -m venv venv
source venv/bin/activate
pip3 install --upgrade pip
pip3 install -r requirements.txt
```

## Запуск проекта
Поправить пути до новых датасетов в [конфигурации](conf/data/sign_train.yaml)
```bash
python3 train.py
```
В репозитории настроено логирование результатов обучения.

```bash
tensorboard --logdir=tensorboard
```
По адресу http://localhost:6006/ вы увидите, как изменяются метрики (accuracy, f1, precision, recall)
