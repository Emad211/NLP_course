import json
import time
import random
from dataclasses import dataclass
from typing import Dict, List

from bs4 import BeautifulSoup

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# ------------------------- تنظیمات -------------------------

@dataclass
class LinkScraperConfig:
    driver_path: str
    chrome_binary: str = None
    headless: bool = False
    scroll_delay: float = 1.5  # مکث بین هر اسکرول

# ------------------------- اسکرپر لینک‌ها -------------------------

class NobatLinkScraper:
    def __init__(self, config: LinkScraperConfig):
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
        
        # مخفی کردن وضعیت WebDriver
        self.driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
            "source": """
                Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
                window.navigator.chrome = {runtime: {}};
            """
        })

    def close(self):
        try:
            self.driver.quit()
        except Exception:
            pass

    def _scroll_to_bottom(self):
        """اسکرول کردن تا انتهای صفحه برای لود شدن تمام پزشکان"""
        last_height = self.driver.execute_script("return document.body.scrollHeight")
        consecutive_scrolls_no_change = 0

        while True:
            # اسکرول به پایین
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            
            # مکث برای لود شدن آیتم‌های جدید
            time.sleep(self.config.scroll_delay + random.uniform(0.5, 1.5))
            
            # بررسی ارتفاع جدید صفحه
            new_height = self.driver.execute_script("return document.body.scrollHeight")
            
            if new_height == last_height:
                # اگر ارتفاع تغییر نکرده، یعنی به انتهای لیست رسیده‌ایم
                consecutive_scrolls_no_change += 1
                if consecutive_scrolls_no_change >= 3:
                    break # ۳ بار پیاپی تغییر نکرد، پس لیست تموم شده
            else:
                consecutive_scrolls_no_change = 0
                last_height = new_height

    def get_specialty_links(self, city_id: int = 1) -> Dict[str, str]:
        """استخراج لینک تمام تخصص‌ها از صفحه اصلی نوبت"""
        base_url = f"https://nobat.ir/find/city-{city_id}/"
        print(f"Fetching specialties from base URL: {base_url}")
        self.driver.get("https://nobat.ir")
        time.sleep(3)  # صبر برای لود کامل منوها

        soup = BeautifulSoup(self.driver.page_source, "html.parser")
        specialties = {}

        # پیدا کردن منوی تخصص‌ها بر اساس HTML ای که فرستادی
        # این پنل در سایت مخفی است اما در سورس کد وجود دارد
        specialty_panel = soup.find("div", {"data-role": "specialty-list-panel"})
        if not specialty_panel:
            # پشتیبان: جستجو در سایدبار دسکتاپ
            specialty_panel = soup.find("aside", {"data-role": "specialty-list"})

        if specialty_panel:
            items = specialty_panel.find_all("li", class_="category-item")
            for item in items:
                a_tag = item.find("a")
                if a_tag and a_tag.get("href"):
                    name = a_tag.get_text(strip=True)
                    href = a_tag["href"]
                    if "/c-" in href:  # مطمئن شدن اینکه لینک یک تخصص است
                        specialties[name] = href
        else:
            print("Error: Could not find specialty list in HTML!")

        print(f"Found {len(specialties)} specialties.")
        return specialties

    def extract_doctor_urls(self, category_url: str) -> List[str]:
        """استخراج تمام لینک‌های پروفایل پزشکان از یک صفحه تخصص"""
        print(f"Opening category: {category_url}")
        self.driver.get(category_url)
        
        # صبر برای لود شدن اولین دکتر
        try:
            WebDriverWait(self.driver, 15).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "a.doctor-ui"))
            )
        except:
            print("No doctors found or page took too long to load.")
            return []

        # اسکرول تا انتهای صفحه
        print("Scrolling to load all doctors...")
        self._scroll_to_bottom()
        print("Finished scrolling. Extracting URLs...")

        # استخراج لینک‌ها از HTML
        soup = BeautifulSoup(self.driver.page_source, "html.parser")
        urls = set() # استفاده از set برای جلوگیری از لینک‌های تکراری

        # بر اساس HTML فرستاده شده، کارت دکترها تگ a با کلاس doctor-ui هستند
        for a_tag in soup.find_all("a", class_="doctor-ui"):
            href = a_tag.get("href")
            if href and href.startswith("https://nobat.ir/"):
                # حذف پارامترهای اضافه مثل ?page=...
                clean_url = href.split("?")[0]
                # حذف اسلش آخر اگر وجود داشت
                clean_url = clean_url.rstrip("/")
                urls.add(clean_url)

        return list(urls)

    def scrape_all_specialties(self, city_id: int = 1, output_file: str = "doctors_urls.json"):
        all_data = {}

        # مرحله 1: گرفتن لیست تخصص‌ها
        specialties = self.get_specialty_links(city_id=city_id)
        
        if not specialties:
            print("No specialties found. Exiting.")
            return

        # مرحله 2: پیمایش هر تخصص
        for specialty_name, url in specialties.items():
            print(f"\n{'='*50}")
            print(f"Scraping Specialty: {specialty_name}")
            print(f"{'='*50}")
            
            try:
                doctor_urls = self.extract_doctor_urls(url)
                all_data[specialty_name] = doctor_urls
                print(f"Found {len(doctor_urls)} doctors for {specialty_name}")
            except Exception as e:
                print(f"Error scraping {specialty_name}: {e}")
                all_data[specialty_name] = []
                
            # استراحت بین هر تخصص برای لو نرفتن
            time.sleep(random.uniform(5, 10))

        # ذخیره در فایل JSON
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(all_data, f, ensure_ascii=False, indent=2)
        
        print(f"\nAll URLs saved to {output_file}")
        return all_data


if __name__ == "__main__":
    # --- تنظیمات ---
    # برای تغییر شهر، شماره شهر را در city_id تغییر دهید (مثلا 1 تهران، 2 مشهد)
    CITY_ID = 1 

    config = LinkScraperConfig(
        driver_path=r"C:\Users\Emad Karimi\Desktop\کرالر نوبت\chromedriver-win64\chromedriver.exe",
        chrome_binary=r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        headless=False # حتما False باشد تا سایت مانع اسکرول شدن نشود
    )

    scraper = NobatLinkScraper(config)
    try:
        scraper.scrape_all_specialties(city_id=CITY_ID, output_file="doctors_urls.json")
    finally:
        scraper.close()