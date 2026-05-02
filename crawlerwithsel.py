import json
import re
import time
import random
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

from bs4 import BeautifulSoup

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# ------------------------- تنظیمات -------------------------

@dataclass
class ScraperConfig:
    driver_path: str
    chrome_binary: Optional[str] = None  
    
    headless: bool = False  

    page_load_timeout: int = 60
    selenium_wait_timeout: int = 30

    min_page_delay: float = 2.0
    max_page_delay: float = 4.0

# ------------------------- ابزارهای کمکی -------------------------

_PERSIAN_ARABIC_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")

def normalize_digits(text: str) -> str:
    if not text:
        return text
    return text.translate(_PERSIAN_ARABIC_DIGITS)

def extract_first_int(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    t = normalize_digits(text)
    m = re.search(r"(\d[\d,]*)", t)
    if not m:
        return None
    return int(m.group(1).replace(",", ""))

def jitter_sleep(a: float, b: float):
    time.sleep(random.uniform(a, b))

# ------------------------- تابع تبدیل استار به لیبل ML -------------------------

def get_sentiment_label(rate_float: float) -> int:
    """
    تبدیل امتیاز (از 5) به 3 کلاس برای یادگیری ماشین:
    -1 -> منفی (1 و 2 ستاره)
     0 -> خنثی (3 ستاره)
     1 -> مثبت (4 و 5 ستاره)
    """
    if rate_float <= 2.0:
        return -1
    elif rate_float <= 3.5:
        return 0
    else:
        return 1

# ------------------------- اسکرپر -------------------------

class NobatDoctorScraperStealth:
    def __init__(self, config: ScraperConfig):
        self.config = config
        self.static_ua = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

        opts = Options()
        
        if self.config.headless:
            opts.add_argument("--headless=new")
            
        opts.add_argument("--window-size=1366,768")
        opts.add_argument("--no-sandbox")
        opts.add_argument("--disable-dev-shm-usage")
        
        opts.add_argument("--disable-blink-features=AutomationControlled")
        opts.add_experimental_option("excludeSwitches", ["enable-automation"])
        opts.add_experimental_option("useAutomationExtension", False)
        opts.add_argument(f"--user-agent={self.static_ua}")
        
        if self.config.chrome_binary:
            opts.binary_location = self.config.chrome_binary

        service = Service(executable_path=self.config.driver_path)
        self.driver = webdriver.Chrome(service=service, options=opts)
        self.driver.set_page_load_timeout(self.config.page_load_timeout)
        
        self.driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
            "source": """
                Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
                Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3, 4, 5]});
                Object.defineProperty(navigator, 'languages', {get: () => ['fa-IR', 'fa', 'en-US', 'en']});
                window.navigator.chrome = {runtime: {}};
            """
        })

        self.wait = WebDriverWait(self.driver, self.config.selenium_wait_timeout)

    def close(self):
        try:
            self.driver.quit()
        except Exception:
            pass

    def _soup(self) -> BeautifulSoup:
        return BeautifulSoup(self.driver.page_source, "html.parser")

    def extract_doctor_id_from_soup(self, soup: BeautifulSoup) -> str:
        el = soup.find(attrs={"data-drid": True})
        if el and el.get("data-drid"):
            return el["data-drid"]
        for script in soup.find_all("script"):
            script_text = script.string or script.get_text()
            if not script_text: continue
            m = re.search(r'data-drid=["\'](\d+)["\']', script_text)
            if m: return m.group(1)
        raise ValueError("Could not extract doctor_id from profile page")

    def extract_rating_percentage(self, style_attr: Optional[str]) -> Optional[float]:
        if not style_attr: return None
        m = re.search(r"width:\s*(\d+(?:\.\d+)?)%", style_attr)
        return float(m.group(1)) if m else None

    def extract_phone_numbers(self, soup: BeautifulSoup) -> List[str]:
        phones = []
        for e in soup.select(".office-phone"):
            t = e.get_text(strip=True)
            if t: phones.append(t)
        return phones

    def extract_offices(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        offices = []
        for office_div in soup.select("div.office[data-officeid]"):
            item = {
                "office_id": office_div.get("data-officeid"), "title": None,
                "address": None, "description": None, "phones": [], "holiday_note": None,
            }
            title = office_div.select_one("strong.office-title")
            if title: item["title"] = title.get_text(strip=True)
            address = office_div.select_one("div.office-address")
            if address: item["address"] = address.get_text(" ", strip=True)
            desc = office_div.select_one("div.office-description")
            if desc: item["description"] = desc.get_text(" ", strip=True)
            holiday = office_div.select_one("div.office-holiday")
            if holiday: item["holiday_note"] = holiday.get_text(strip=True)
            for p in office_div.select("div.office-phone"):
                pt = p.get_text(strip=True)
                if pt: item["phones"].append(pt)
            offices.append(item)
        return offices

    def extract_social_links(self, soup: BeautifulSoup) -> Dict[str, str]:
        socials = {}
        for a in soup.select("a[href]"):
            href = (a.get("href") or "").strip()
            if not href: continue
            if "instagram.com" in href: socials["instagram"] = href
            elif "t.me" in href or "telegram.me" in href: socials["telegram"] = href
            elif "whatsapp.com" in href or "wa.me" in href: socials["whatsapp"] = href
            elif "linkedin.com" in href: socials["linkedin"] = href
        return socials

    def open_profile(self, profile_url: str):
        self.driver.get(profile_url)
        self.wait.until(
            EC.any_of(
                EC.presence_of_element_located((By.CSS_SELECTOR, "h1.doctor-ui-name")),
                EC.presence_of_element_located((By.CSS_SELECTOR, "div#doctor")),
                EC.presence_of_element_located((By.CSS_SELECTOR, "[data-drid]")),
            )
        )
        jitter_sleep(self.config.min_page_delay, self.config.max_page_delay)

    def scrape_profile(self, profile_url: str) -> Dict[str, Any]:
        self.open_profile(profile_url)
        soup = self._soup()
        doctor_id = self.extract_doctor_id_from_soup(soup)
        data = {
            "doctor_id": doctor_id, "profile_url": profile_url, "nice_id": None,
            "name": None, "specialty": None, "medical_code": None, "verified": False,
            "followers_count": None, "reviews_count": None, "rating_percent": None,
            "rating_score_5": None, "biography": None, "social_media": {},
            "phones": [], "offices": [], "schedule": []
        }
        doctor_div = soup.find("div", id="doctor")
        if doctor_div: data["nice_id"] = doctor_div.get("data-niceid")
        name = soup.select_one("h1.doctor-ui-name")
        if name: data["name"] = name.get_text(strip=True)
        specialty = soup.select_one("h2.doctor-ui-specialty")
        if specialty: data["specialty"] = specialty.get_text(strip=True)
        code = soup.select_one("div.doctor-code")
        if code: data["medical_code"] = code.get_text(strip=True)
        if soup.select_one(".doctor-ui-profile .verified"): data["verified"] = True
        followers = soup.select_one(".followers-title")
        if followers: data["followers_count"] = extract_first_int(followers.get_text(strip=True))
        reviews = soup.select_one(".comments-summary")
        if reviews: data["reviews_count"] = extract_first_int(reviews.get_text(strip=True))
        stars_value = soup.select_one(".stars-value")
        if stars_value:
            percent = self.extract_rating_percentage(stars_value.get("style"))
            data["rating_percent"] = percent
            if percent is not None: data["rating_score_5"] = round((percent / 100.0) * 5.0, 2)
        bio = soup.select_one("div.doctor-bio-text")
        if bio: data["biography"] = bio.get_text(" ", strip=True)
        data["social_media"] = self.extract_social_links(soup)
        data["phones"] = self.extract_phone_numbers(soup)
        data["offices"] = self.extract_offices(soup)
        return data

    def get_all_comments_via_ui(self) -> List[Dict[str, Any]]:
        # اسکرول به بخش کامنت‌ها
        try:
            comments_section = self.driver.find_element(By.CSS_SELECTOR, ".comments, #comments, .comments-summary")
            self.driver.execute_script("arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});", comments_section)
            jitter_sleep(2, 3)
        except Exception:
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            jitter_sleep(2, 3)

        # کلیک روی دکمه‌های لود بیشتر
        last_count = 0
        retry = 0
        while True:
            buttons = self.driver.find_elements(By.CSS_SELECTOR, "button.load-more, button.btn-more, a.load-more, button.show-more, button, a.more-comments")
            clicked = False
            
            for btn in buttons:
                try:
                    text = btn.text.strip()
                    if btn.is_displayed() and ("بیشتر" in text or "مشاهده بیشتر" in text or "نظر بیشتر" in text or "load more" in text.lower()):
                        self.driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", btn)
                        jitter_sleep(0.5, 1.0)
                        btn.click()
                        clicked = True
                        jitter_sleep(2.0, 3.0)
                        break
                except Exception:
                    continue
            
            soup_temp = self._soup()
            current_count = len(soup_temp.select("div.comment, div.comment-item, div.comment-box, div.review-item"))
            
            if current_count > last_count:
                last_count = current_count
                retry = 0
            else:
                if not clicked: break
                retry += 1
                if retry >= 3: break

        # استخراج اطلاعات از HTML نهایی
        soup = self._soup()
        all_comments = []
        
        comment_elements = soup.select("div.comment, div.comment-item, div.comment-box, div.review-item")
        
        for el in comment_elements:
            text_el = el.select_one(".comment-text, .text, p, .comment-body, .comment-content")
            date_el = el.select_one(".comment-date, .date, time, span.date")
            
            # استخراج امتیاز (Rate)
            rate_float = None
            stars_value_el = el.select_one(".stars-value, .star-rating-value, .fill")
            if stars_value_el and stars_value_el.get("style"):
                percent = self.extract_rating_percentage(stars_value_el.get("style"))
                if percent is not None:
                    rate_float = round((percent / 100.0) * 5, 1)

            like_val, dislike_val = 0, 0
            like_el = el.select_one(".like, .like-count, .btn-like span")
            dislike_el = el.select_one(".dislike, .dislike-count, .btn-dislike span")
            if like_el:
                temp = extract_first_int(like_el.get_text(strip=True))
                if temp is not None: like_val = temp
            if dislike_el:
                temp = extract_first_int(dislike_el.get_text(strip=True))
                if temp is not None: dislike_val = temp

            if text_el:
                text = text_el.get_text(" ", strip=True)
                
                # --- فیلتر کردن برای دیتاست یادگیری ماشین ---
                # کامنت‌های بدون امتیاز یا بدون متن برای ML نویز هستند
                if text and len(text) > 1 and rate_float is not None:
                    
                    # محاسبه لیبل بر اساس امتیاز
                    label = get_sentiment_label(rate_float)
                    
                    all_comments.append({
                        "text": text,
                        "rate": rate_float,
                        "label": label,   # لیبل شما: -1, 0, 1
                        "date": date_el.get_text(strip=True) if date_el else None,
                        "like": like_val,
                        "dislike": dislike_val
                    })
                    
        return all_comments

    def scrape_doctor(self, profile_url: str, output_file: str = "doctor_complete.json") -> Dict[str, Any]:
        print(f"Start stealth scraping: {profile_url}")
        profile = self.scrape_profile(profile_url)
        print(f"Doctor ID: {profile['doctor_id']}\nName: {profile.get('name')}")
        
        comments = self.get_all_comments_via_ui()
        
        # شمارش تعداد لیبل‌ها برای گزارش
        pos = sum(1 for c in comments if c['label'] == 1)
        neu = sum(1 for c in comments if c['label'] == 0)
        neg = sum(1 for c in comments if c['label'] == -1)
        
        print(f"Total ML-Ready comments: {len(comments)}")
        print(f"Label Distribution -> Positive(1): {pos}, Neutral(0): {neu}, Negative(-1): {neg}")
        
        result = {"doctor_profile": profile, "comments": comments}
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"Saved to {output_file}")
        return result


if __name__ == "__main__":
    config = ScraperConfig(
        driver_path=r"C:\Users\Emad Karimi\Desktop\کرالر نوبت\chromedriver-win64\chromedriver.exe",
        chrome_binary=r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        headless=False
    )

    scraper = NobatDoctorScraperStealth(config)
    
    try:
        # خواندن فایلی که اسکرپر لینک‌ساز ساخته است
        with open("doctors_urls.json", "r", encoding="utf-8") as f:
            all_specialties = json.load(f)
            
        total_doctors = sum(len(urls) for urls in all_specialties.values())
        current_count = 0

        # حلقه روی هر تخصص
        for specialty, urls in all_specialties.items():
            print(f"\n{'#'*60}")
            print(f"STARTING SPECIALTY: {specialty} ({len(urls)} doctors)")
            print(f"{'#'*60}\n")
            
            # حلقه روی دکترهای این تخصص
            for profile_url in urls:
                current_count += 1
                # نام فایل را با نام تخصص ترکیب می‌کنیم تا مرتب بشود
                safe_specialty = specialty.replace(" ", "_").replace("/", "_")
                output_file = f"{safe_specialty}_doctor_{current_count}.json"
                
                print(f"Progress: {current_count}/{total_doctors}")
                try:
                    scraper.scrape_doctor(profile_url, output_file=output_file)
                except Exception as e:
                    print(f"Error scraping {profile_url}: {e}")
                
                # استراحت بین هر دکتر برای لو نرفتن
                jitter_sleep(3.0, 6.0)

    finally:
        scraper.close()