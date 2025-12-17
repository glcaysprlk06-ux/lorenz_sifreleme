
"""
AES-GCM + SQLite entegrasyonu.
Deprem CSV kaydından deterministik Lorenz tabanlı anahtar türetme.

Kullanım:
  - quakes.csv dosyasında şu kolonlar bulunmalı: id, latitude, longitude, time (UNIX saniye)
  - Bu kodu çalıştır ve şifreleme / çözme seçeneklerini takip et.

Gereksinimler:
  pip install numpy scipy cryptography
"""

import csv
import hashlib
import struct
import os
import sqlite3
import numpy as np
from datetime import datetime
from scipy.integrate import solve_ivp
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

# ---------- Ayarlar ----------
QUAKES_CSV = "quakes.csv"
DB_PATH = "quake_messages_clean.db"

# Kullanıcının seçebileceği hash algoritmaları
HASH_ALGS = {1: "sha256", 2: "md5", 3: "sha1"}  # güvenlik için 1 (sha256) önerilir

# ---------- Deprem CSV yükleme ----------
def load_quakes_csv(path):
    """Deprem verilerini şu formatta liste olarak yükler: [{'id','lat','lon','time'}, ...]"""
    quakes = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                lat = float(row["latitude"])
                lon = float(row["longitude"])
                t = float(row["time"])
            except Exception:
                continue
            quakes.append({"id": row.get("id"), "lat": lat, "lon": lon, "time": t})
    return quakes

# ---------- Bir deprem kaydından hash üret ----------
def compute_hash_for_record(record, hash_choice=1):
    """
    Deprem kaydından deterministik bir hash üretir.
    record: {'lat','lon','time'}
    """
    time_int = int(record["time"])  # kayan nokta hatalarını engellemek için
    s = f"{record['lat']}|{record['lon']}|{time_int}"
    alg = HASH_ALGS.get(hash_choice, "sha256")
    h = hashlib.new(alg, s.encode("utf-8")).digest()
    return h

# ---------- Hash → Lorenz başlangıç durumu ----------
def hash_to_state(hash_bytes):
    """
    Hash verisini Lorenz denklemi için 3 başlangıç değişkenine çevirir.
    Dönen: [x0, y0, z0]
    """
    if len(hash_bytes) < 24:
        hash_bytes = (hash_bytes * (24 // len(hash_bytes) + 1))[:24]
    parts = [hash_bytes[i*8:(i+1)*8] for i in range(3)]
    vals = []
    for p in parts:
        u = int.from_bytes(p, byteorder="big", signed=False)
        f = 0.1 + (u / (2**64 - 1)) * 0.9  # 0.1 ile 1.0 arası
        vals.append(float(f))
    return vals

# ---------- Lorenz + Anahtar Türetme ----------
def lorenz(t, state, sigma, rho, beta):
    x, y, z = state
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return [dx, dy, dz]

def derive_key_from_state(state, sigma=10.0, rho=28.0, beta=8/3, t_span=(0.0, 2.0), t_steps=3000):
    """
    Deterministik Anahtar Türetme:
      - Lorenz ODE çözümü
      - Tüm noktaları double formatında paketle
      - SHA-256 ile 32 baytlık AES anahtarı üret
    """
    t_eval = np.linspace(t_span[0], t_span[1], t_steps)
    sol = solve_ivp(lorenz, t_span, state, args=(sigma, rho, beta),
                    t_eval=t_eval, rtol=1e-9, atol=1e-12)
    traj = sol.y.flatten()
    b = b"".join(struct.pack(">d", float(v)) for v in traj)
    key = hashlib.sha256(b).digest()
    return key

# ---------- AES-GCM Yardımcıları ----------
def aes_gcm_encrypt(key: bytes, plaintext: bytes, aad: bytes = None):
    aesgcm = AESGCM(key)
    nonce = os.urandom(12)  # 96-bit nonce önerilir
    ct_and_tag = aesgcm.encrypt(nonce, plaintext, aad)
    tag = ct_and_tag[-16:]
    ciphertext = ct_and_tag[:-16]
    return nonce, ciphertext, tag

def aes_gcm_decrypt(key: bytes, nonce: bytes, ciphertext: bytes, tag: bytes, aad: bytes = None):
    aesgcm = AESGCM(key)
    ct_and_tag = ciphertext + tag
    pt = aesgcm.decrypt(nonce, ct_and_tag, aad)
    return pt

# ---------- SQLite Veritabanı ----------
def init_db(path=DB_PATH):
    conn = sqlite3.connect(path)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nonce BLOB NOT NULL,
            ciphertext BLOB NOT NULL,
            tag BLOB NOT NULL,
            quake_index INTEGER NOT NULL,
            hash_choice INTEGER NOT NULL,
            created TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def save_record(nonce: bytes, ciphertext: bytes, tag: bytes, quake_index: int, hash_choice: int, path=DB_PATH):
    conn = sqlite3.connect(path)
    c = conn.cursor()
    created = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO messages (nonce, ciphertext, tag, quake_index, hash_choice, created) VALUES (?, ?, ?, ?, ?, ?)",
              (nonce, ciphertext, tag, quake_index, hash_choice, created))
    rec_id = c.lastrowid
    conn.commit()
    conn.close()
    return rec_id

def fetch_record_by_id(rec_id: int, path=DB_PATH):
    conn = sqlite3.connect(path)
    c = conn.cursor()
    c.execute("SELECT id, nonce, ciphertext, tag, quake_index, hash_choice, created FROM messages WHERE id = ?", (rec_id,))
    row = c.fetchone()
    conn.close()
    return row

def fetch_record_by_ciphertext_hex(ct_hex: str, path=DB_PATH):
    """
    Kullanıcının girdiği hex'i hem sadece ciphertext olarak
    hem de ciphertext+tag olarak kontrol eder.
    """
    try:
        candidate = bytes.fromhex(ct_hex)
    except Exception:
        return None

    conn = sqlite3.connect(path)
    c = conn.cursor()

    # Sadece ciphertext eşleşirse
    c.execute("SELECT id, nonce, ciphertext, tag, quake_index, hash_choice, created FROM messages WHERE ciphertext = ?", (candidate,))
    row = c.fetchone()
    if row:
        conn.close()
        return row

    # Hem ciphertext hem tag olabilir
    if len(candidate) > 16:
        cand_ct = candidate[:-16]
        cand_tag = candidate[-16:]
        c.execute("SELECT id, nonce, ciphertext, tag, quake_index, hash_choice, created FROM messages WHERE ciphertext = ? AND tag = ?", (cand_ct, cand_tag))
        row = c.fetchone()
        conn.close()
        return row

    conn.close()
    return None

# ---------- Yüksek seviye işlemler ----------
def encrypt_and_store(quakes, quake_index: int, hash_choice: int, plaintext: bytes, aad: bytes = None):
    if quake_index < 1 or quake_index > len(quakes):
        raise ValueError("quake_index geçerli aralıkta değil")
    record = quakes[quake_index - 1]
    h = compute_hash_for_record(record, hash_choice=hash_choice)
    state = hash_to_state(h)
    key = derive_key_from_state(state)
    nonce, ciphertext, tag = aes_gcm_encrypt(key, plaintext, aad=aad)
    rec_id = save_record(nonce, ciphertext, tag, quake_index, hash_choice)
    return rec_id, nonce, ciphertext, tag

def decrypt_record_row(row, quakes, aad: bytes = None):
    if row is None:
        raise ValueError("Kayıt bulunamadı")
    _id, nonce, ciphertext, tag, quake_index, hash_choice, created = row
    if quake_index < 1 or quake_index > len(quakes):
        raise ValueError("quake_index geçersiz")
    record = quakes[quake_index - 1]
    h = compute_hash_for_record(record, hash_choice=hash_choice)
    state = hash_to_state(h)
    key = derive_key_from_state(state)
    plaintext = aes_gcm_decrypt(key, nonce, ciphertext, tag, aad=aad)
    return plaintext

# ---------- Komut satırı arabirimi ----------
def main():
    quakes = load_quakes_csv(QUAKES_CSV)
    if not quakes:
        print("quakes.csv bulunamadı veya boş. Dosyada id, latitude, longitude, time sütunları olmalı.")
        return

    init_db()
    print("Seçenekler:")
    print("1) Şifrele ve kaydet (deprem kaydı + hash seçimi)")
    print("2) Kayıt ID ile çöz")
    print("3) Şifrelenmiş metin (hex) ile çöz")

    choice = input("Seç: ").strip()

    if choice == "1":
        qi = int(input(f"Deprem indexi (1..{len(quakes)}): ").strip())
        print("Hash seçenekleri:", HASH_ALGS)
        hc = int(input("hash_choice (örn: 1 = sha256): ").strip())

        print("Şifrelenecek metni yaz (çok satırlı, bitirmek için boş satır):")
        lines = []
        while True:
            ln = input()
            if ln == "":
                break
            lines.append(ln)
        plaintext = "\n".join(lines).encode("utf-8")

        aad_input = input("Opsiyonel AAD (boş bırakılabilir): ").strip()
        aad = aad_input.encode("utf-8") if aad_input else None

        rec_id, nonce, ct, tag = encrypt_and_store(quakes, qi, hc, plaintext, aad=aad)
        print("Kayıt ID:", rec_id)
        print("Nonce (hex):", nonce.hex())
        print("Ciphertext (hex):", ct.hex())
        print("Tag (hex):", tag.hex())
        print("Bu üç hex değeri (nonce, ciphertext, tag) deşifre etmek için yeterlidir.")

    elif choice == "2":
        rid = int(input("Kayıt ID: ").strip())
        row = fetch_record_by_id(rid)
        if not row:
            print("Bu ID'de kayıt yok")
            return

        aad_input = input("Şifrelemede kullanılan AAD (yoksa boş bırak): ").strip()
        aad = aad_input.encode("utf-8") if aad_input else None

        try:
            pt = decrypt_record_row(row, quakes, aad=aad)
            print("Çözülen metin:\n")
            print(pt.decode("utf-8"))
        except Exception as e:
            print("Çözme hatası:", e)

    elif choice == "3":
        hex_ct = input("Ciphertext hex (SADECE ciphertext veya ciphertext+tag): ").strip()
        row = fetch_record_by_ciphertext_hex(hex_ct)
        if not row:
            print("Eşleşen kayıt bulunamadı")
            return

        aad_input = input("Şifrelemede kullanılan AAD (yoksa boş bırak): ").strip()
        aad = aad_input.encode("utf-8") if aad_input else None

        try:
            pt = decrypt_record_row(row, quakes, aad=aad)
            print("Çözülen metin:\n")
            print(pt.decode("utf-8"))
        except Exception as e:
            print("Çözme hatası:", e)

    else:
        print("Geçersiz seçim")

if __name__ == "__main__":
    main()
