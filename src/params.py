"""Разбор текстовых параметров: «Вид услуги X», «Тип услуги Y», фильтр по рейтингу.

В параметрах запроса лежат фильтры, которые пользователь выставил в поиске. Если стоит фильтр
«Вид услуги Ремонт и отделка», у выбранного объявления почти наверняка тот же вид услуги.
"""
import re

# Ключи фильтров в параметрах запроса: значение тянется до следующего ключа или конца строки.
_Q_KEYS = (r"(?= Тип услуги| Вид услуги| Кто оказывает| Онлайн-запись| Рейтинг| Срочная| Где вы| Сортировка"
           r"| Гарантия| Поиск по| Ваши клиенты| Услуга|$)")
_VID_Q = re.compile(r"Вид услуги (.+?)" + _Q_KEYS)
# «Тип услуги автосервиса ...» — отдельный ключ (подтип внутри автосервиса), это не значение «Тип услуги»
_TIP_Q = re.compile(r"Тип услуги (?!автосервиса )(.+?)" + _Q_KEYS)


def parse_query_params(s: str) -> dict:
    """Фильтры запроса. Видов услуги в запросе не бывает больше одного, типов — может быть несколько,
    поэтому q_tips хранит их все через «|»."""
    s = s or ""
    vid = [m.group(1).strip() for m in _VID_Q.finditer(s)]
    tips = sorted({m.group(1).strip() for m in _TIP_Q.finditer(s)})
    return {
        "q_vid": vid[0] if vid else "",
        "q_tips": "|".join(tips),
        "q_rating4": int("Рейтинг пользователя 4 звезды" in s),
    }


def _values_regex(key, values):
    """Регулярка «<ключ> <одно из известных значений>». Длинные значения первыми, чтобы
    «Ремонт и отделка» не обрезалось до «Ремонт»."""
    values = sorted({v for v in values if v}, key=len, reverse=True)
    return re.compile(key + r" (" + "|".join(map(re.escape, values)) + r")(?= |$)")


def build_item_parsers(vid_values, tip_values):
    """Парсеры параметров объявления. У объявлений ключей слишком много, чтобы резать строку по ключам,
    поэтому значения ищем по словарю известных значений, собранному из фильтров запросов."""
    vid_re = _values_regex("Вид услуги", vid_values)
    tip_re = _values_regex("Тип услуги", tip_values)

    def item_vid(s: str) -> str:
        m = vid_re.search(s or "")
        return m.group(1) if m else ""

    def item_tips(s: str) -> str:
        return "|".join(sorted(set(tip_re.findall(s or ""))))

    return item_vid, item_tips
