from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from random import choices
from string import digits, ascii_lowercase
import dateparser
import os
import time

weekdays_en = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
weekdays_zh = ["一", "二", "三", "四", "五", "六", "日"]
months_en = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]

def now():
    return datetime.now(ZoneInfo("Asia/Shanghai"))

def hours_ago(hours):
    return now() - timedelta(hours=hours)

def hours_later(hours):
    return now() + timedelta(hours=hours)

def days_ago(days):
    return now() - timedelta(days=days)

def days_later(days):
    return now() + timedelta(days=days)

def today_iso():
    return now().date().isoformat()

def now_and_choices():
    return now().strftime("%Y%m%d%H%M%S") + "".join(choices(digits + ascii_lowercase, k=4))

def get_current_hour():
    return now().hour

def get_day_suffix(day):
    return "th" if 11 <= day <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")

def get_current_day_en():
    return f"{now().day}{get_day_suffix(now().day)}"

def get_current_day_zh():
    return f"{now().day}日"

def get_previous_day_en():
    return f"{(now() - timedelta(days=1)).day}{get_day_suffix((now() - timedelta(days=1)).day)}"

def get_previous_day_zh():
    return f"{(now() - timedelta(days=1)).day}日"

def get_next_day_en():
    return f"{(now() + timedelta(days=1)).day}{get_day_suffix((now() + timedelta(days=1)).day)}"

def get_next_day_zh():
    return f"{(now() + timedelta(days=1)).day}日"

def get_current_weekday_en():
    return weekdays_en[now().weekday()]

def get_current_weekday_zh():
    return f"星期{weekdays_zh[now().weekday()]}"

def get_previous_weekday_en():
    return weekdays_en[(now().weekday() - 1) % 7]

def get_previous_weekday_zh():
    return f"星期{weekdays_zh[(now().weekday() - 1) % 7]}"

def get_next_weekday_en():
    return weekdays_en[(now().weekday() + 1) % 7]

def get_next_weekday_zh():
    return f"星期{weekdays_zh[(now().weekday() + 1) % 7]}"

def get_current_month_en():
    return months_en[now().month - 1]

def get_current_month_zh():
    return f"{now().month}月"

def get_previous_month_en():
    return months_en[(now().month - 2) % 12]

def get_previous_month_zh():
    return f"{(now().month - 1) or 12}月"

def get_next_month_en():
    return months_en[now().month % 12]

def get_next_month_zh():
    return f"{(now().month % 12) + 1}月"

def get_current_year():
    return f"{now().year}"

def get_previous_year():
    return f"{now().year - 1}"

def get_next_year():
    return f"{now().year + 1}"

def get_today_with_weekday_en():
    today = now()
    return f"{months_en[today.month - 1]} {today.day}{get_day_suffix(today.day)}, {weekdays_en[today.weekday()]}"

def get_today_with_weekday_zh():
    today = now()
    return f"{today.month}月{today.day}日，星期{weekdays_zh[today.weekday()]}"

def get_yesterday_with_weekday_en():
    yesterday = now() - timedelta(days=1)
    return f"{months_en[yesterday.month - 1]} {yesterday.day}{get_day_suffix(yesterday.day)}, {weekdays_en[yesterday.weekday()]}"

def get_yesterday_with_weekday_zh():
    yesterday = now() - timedelta(days=1)
    return f"{yesterday.month}月{yesterday.day}日，星期{weekdays_zh[yesterday.weekday()]}"

def get_tomorrow_with_weekday_en():
    tomorrow = now() + timedelta(days=1)
    return f"{months_en[tomorrow.month - 1]} {tomorrow.day}{get_day_suffix(tomorrow.day)}, {weekdays_en[tomorrow.weekday()]}"

def get_tomorrow_with_weekday_zh():
    tomorrow = now() + timedelta(days=1)
    return f"{tomorrow.month}月{tomorrow.day}日，星期{weekdays_zh[tomorrow.weekday()]}"

def get_weekdays_with_dates_en():
    today = now()
    start_of_week = today - timedelta(days=today.weekday())
    dates = [start_of_week + timedelta(days=i) for i in range(-7, 14)]
    return "; ".join([", ".join([f"{'this' if i == 7 else 'last' if i == 0 else 'next'} {weekdays_en[d.weekday()]} is {months_en[d.month - 1]} {d.day}{get_day_suffix(d.day)}" for d in dates[i:i + 7]]) for i in (7, 0, 14)])

def get_weekdays_with_dates_zh():
    today = now()
    start_of_week = today - timedelta(days=today.weekday())
    dates = [start_of_week + timedelta(days=i) for i in range(-7, 14)]
    return "；".join(["，".join([f"{'本' if i == 7 else '上' if i == 0 else '下'}周{weekdays_zh[d.weekday()]}是{d.month}月{d.day}日" for d in dates[i:i + 7]]) for i in (7, 0, 14)])

def get_recent_dates_iso(days):
    return [(now() - timedelta(days=i)).date().isoformat() for i in range(days)]

def iso_date(timestamp):
    try:
        if isinstance(timestamp, (int, float)):
            dt = datetime.fromtimestamp(timestamp / 1000 if timestamp > 1e10 else timestamp, ZoneInfo("Asia/Shanghai"))
        else:
            dt = dateparser.parse(timestamp, settings={"RETURN_AS_TIMEZONE_AWARE": True, "TIMEZONE": "Asia/Shanghai"})
        return dt.date().isoformat()
    except:
        return timestamp

def year_start(year):
    return (datetime(year, 1, 1, tzinfo=ZoneInfo("Asia/Shanghai"))
            if year <= now().year
            else datetime(now().year, 1, 1, tzinfo=ZoneInfo("Asia/Shanghai")))

def year_end(year):
    return (datetime(year, 12, 31, 23, 59, 59, 999999, tzinfo=ZoneInfo("Asia/Shanghai"))
            if year < now().year
            else now())

def scheduled_run(start_hour, start_minute, function):
    try:
        with open("lock", "r") as f:
            os.kill(int(f.read().strip()), 0)
        return
    except Exception:
        with open("lock", "w") as f:
            f.write(str(os.getpid()))
        next_run = now().replace(hour=start_hour, minute=start_minute, second=0, microsecond=0)
        if now() > next_run:
            next_run += timedelta(days=1)
        while True:
            time.sleep(max(0, (next_run - now()).total_seconds()))
            function()
            next_run += timedelta(days=1)

def interval_run(interval_seconds, function):
    while True:
        time.sleep(interval_seconds)
        function()
