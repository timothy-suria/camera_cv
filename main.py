import cv2
import numpy as np
import os
import re
import subprocess
import threading
import time
from collections import deque, Counter
from insightface.app import FaceAnalysis
from insightface.app.common import Face

try:
    import winsound              
except ImportError:
    winsound = None

KNOWN_FACES_DIR = "known_faces"
SIMILARITY_THRESHOLD = 0.45

IMAGE_EXTS = (".jpg", ".jpeg", ".png")
PHOTOS_PER_PERSON = 5                   
TOP_K_MATCH = 1                         

CAM_USERNAME = "admin"
CAM_PASSWORD = "Denpasar2003*"          
CAM_IP = "10.102.100.150"               
CAM_PORT = 554
CAM_CHANNEL = "102"                     
RTSP_URL = f"rtsp://{CAM_USERNAME}:{CAM_PASSWORD}@{CAM_IP}:{CAM_PORT}/Streaming/Channels/{CAM_CHANNEL}"
RTSP_RECONNECT_DELAY = 3.0

# Batasi waktu tunggu koneksi RTSP (mikrodetik) supaya kalau CCTV tidak
# terjangkau, cepat gagal dan langsung fallback ke webcam, bukan menggantung.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp|stimeout;5000000")

WEBCAM_FALLBACK = True
WEBCAM_INDEX = 0

DET_SIZE = 320
REC_INTERVAL = 10

IOU_THRESHOLD = 0.3
MAX_MISSED_FRAMES = 15
VOTE_HISTORY = 10
BOX_SMOOTHING = 0.6

PREVIEW_SIZE = 96
SHOW_SIDE_PANEL = True

WINDOW_NAME = "Face Recognition (CCTV) - 'q' keluar, 'p' panel, 's' suara, 'f' fullscreen"
START_FULLSCREEN = True                  

ALERT_ENABLED = True
ALERT_TEXT = "Perhatian. Intruder tidak dikenal memasuki area kerja."
ALERT_WAV = os.path.join("assets", "alert_intruder.wav")
ALERT_VOICE_PREFIX = "id"                
ALERT_VOTES = 2                          
ALERT_COOLDOWN = 8.0                     
ALERT_REPEAT = 20.0                      


class CameraStream:
    """Ambil frame dari CCTV (RTSP) di thread sendiri supaya loop utama tidak
    blocking nunggu network/decode, dan otomatis reconnect kalau stream putus."""

    def __init__(self, sources):
        # sources: sumber yang dicoba berurutan, mis. [RTSP_URL, 0].
        # String dianggap URL RTSP, integer dianggap index webcam lokal.
        self.sources = list(sources) if isinstance(sources, (list, tuple)) else [sources]
        self.cap = None
        self.active_source = None
        self.frame = None
        self.running = False
        self.lock = threading.Lock()
        self.thread = None
        self._open_capture()

    def _open_capture(self):
        for source in self.sources:
            if isinstance(source, str):
                cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
            else:
                cap = cv2.VideoCapture(source)

            if cap.isOpened():
                if isinstance(source, str):
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self.cap = cap
                if source != self.active_source:
                    label = "RTSP" if isinstance(source, str) else f"webcam #{source}"
                    print(f"[INFO] Sumber video aktif: {label}")
                self.active_source = source
                return
            cap.release()

        self.cap = None

    def is_opened(self):
        return self.cap is not None and self.cap.isOpened()

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

        for _ in range(150):  
            if self.frame is not None:
                break
            time.sleep(0.05)

        return self.frame is not None

    def _loop(self):
        while self.running:
            if not self.is_opened():
                print("[WARNING] Koneksi RTSP terputus, mencoba reconnect...")
                if self.cap is not None:
                    self.cap.release()
                time.sleep(RTSP_RECONNECT_DELAY)
                self._open_capture()
                continue

            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue

            with self.lock:
                self.frame = frame

    def read(self):
        """Ambil frame terbaru (selalu yang paling baru, frame lama dibuang)."""
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        if self.cap is not None:
            self.cap.release()


class IntruderAlert:
    """Peringatan suara waktu ada wajah tak dikenal.

    Kalimatnya dirender sekali jadi file WAV lewat PowerShell (tidak perlu
    install library apa pun), lalu diputar secara async supaya loop kamera tidak
    pernah berhenti nunggu audio selesai.

    TTS-nya dicoba dua jalur, berurutan:
      1. WinRT (Windows.Media.SpeechSynthesis) -- bisa memakai suara OneCore
         seperti "Microsoft Andika" (id-ID), jadi kalimat Indonesia terdengar
         natural.
      2. System.Speech / SAPI5 -- cuma melihat suara "Desktop" (umumnya en-US
         saja), dipakai kalau jalur WinRT gagal.
    Kalau dua-duanya gagal, jatuh ke bunyi beep supaya tetap ada peringatan."""

    def __init__(self, text=ALERT_TEXT, wav_path=ALERT_WAV,
                 voice_prefix=ALERT_VOICE_PREFIX, cooldown=ALERT_COOLDOWN):
        self.wav_path = os.path.abspath(wav_path)
        self.cooldown = cooldown
        self.last_played = 0.0
        self.muted = False
        self.lock = threading.Lock()
        self.ready = self._ensure_wav(text, voice_prefix)

        if not self.ready:
            print("[WARNING] Suara peringatan tidak bisa dibuat, pakai bunyi beep.")

    def _ensure_wav(self, text, voice_prefix):
        """Render WAV kalau belum ada atau kalimat/suaranya berubah."""
        if winsound is None:
            return False

        marker = os.path.splitext(self.wav_path)[0] + ".txt"
        stamp = f"{voice_prefix}|{text}"

        if os.path.exists(self.wav_path) and os.path.exists(marker):
            try:
                with open(marker, encoding="utf-8") as f:
                    if f.read() == stamp and os.path.getsize(self.wav_path) > 0:
                        return True
            except OSError:
                pass

        folder = os.path.dirname(self.wav_path)
        if folder:
            os.makedirs(folder, exist_ok=True)

        rendered = (self._render_winrt(text, voice_prefix)
                    or self._render_sapi(text, voice_prefix))
        if not rendered:
            return False

        try:
            with open(marker, "w", encoding="utf-8") as f:
                f.write(stamp)
        except OSError:
            pass

        print(f"[INFO] Suara peringatan siap ({rendered}): {self.wav_path}")
        return True

    def _run_powershell(self, script):
        """Jalankan script PowerShell tanpa memunculkan jendela. -> CompletedProcess|None"""
        try:
            return subprocess.run(
                ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script],
                capture_output=True, text=True, timeout=90,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except (OSError, subprocess.SubprocessError) as exc:
            print(f"[WARNING] Gagal menjalankan PowerShell: {exc}")
            return None

    def _render_winrt(self, text, voice_prefix):
        """TTS lewat WinRT, bisa pakai suara OneCore (mis. Andika id-ID). -> str|None"""
        safe_text = text.replace("'", "''")
        safe_path = self.wav_path.replace("'", "''")
        safe_prefix = voice_prefix.replace("'", "''")

        script = (
            "$ErrorActionPreference = 'Stop'; "
            "[Windows.Media.SpeechSynthesis.SpeechSynthesizer, Windows.Media, ContentType=WindowsRuntime] | Out-Null; "
            "[Windows.Storage.Streams.DataReader, Windows.Storage.Streams, ContentType=WindowsRuntime] | Out-Null; "
            "Add-Type -AssemblyName System.Runtime.WindowsRuntime; "
            "$asTask = ([System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object { "
            "$_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 -and "
            "$_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1' })[0]; "
            "function Await($op, $type) { "
            "$t = $asTask.MakeGenericMethod($type).Invoke($null, @($op)); "
            "$t.Wait(-1) | Out-Null; $t.Result }; "
            "$voice = [Windows.Media.SpeechSynthesis.SpeechSynthesizer]::AllVoices | "
            f"Where-Object {{ $_.Language -like '{safe_prefix}*' }} | Select-Object -First 1; "
            f"if (-not $voice) {{ throw 'Tidak ada suara dengan bahasa {safe_prefix}' }}; "
            "$synth = New-Object Windows.Media.SpeechSynthesis.SpeechSynthesizer; "
            "$synth.Voice = $voice; "
            f"$stream = Await ($synth.SynthesizeTextToStreamAsync('{safe_text}')) "
            "([Windows.Media.SpeechSynthesis.SpeechSynthesisStream]); "
            "$size = [uint32]$stream.Size; "
            "$reader = New-Object Windows.Storage.Streams.DataReader($stream.GetInputStreamAt(0)); "
            "Await ($reader.LoadAsync($size)) ([uint32]) | Out-Null; "
            "$bytes = New-Object byte[] $size; $reader.ReadBytes($bytes); "
            f"[IO.File]::WriteAllBytes('{safe_path}', $bytes); "
            "$reader.Dispose(); $stream.Dispose(); $synth.Dispose(); "
            "Write-Output $voice.DisplayName"
        )

        result = self._run_powershell(script)
        if result is None:
            return None
        if result.returncode != 0 or not os.path.exists(self.wav_path):
            print(f"[INFO] TTS WinRT tidak bisa dipakai: {result.stderr.strip()[:200]}")
            return None
        return result.stdout.strip() or "WinRT"

    def _render_sapi(self, text, voice_prefix):
        """TTS lewat System.Speech/SAPI5. Cadangan kalau WinRT gagal. -> str|None"""
        safe_text = text.replace("'", "''")
        safe_path = self.wav_path.replace("'", "''")
        safe_prefix = voice_prefix.replace("'", "''")

        script = (
            "Add-Type -AssemblyName System.Speech; "
            "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            "$v = $s.GetInstalledVoices() | Where-Object { "
            f"$_.VoiceInfo.Culture.Name -like '{safe_prefix}*' }} | Select-Object -First 1; "
            "if ($v) { $s.SelectVoice($v.VoiceInfo.Name) }; "
            f"$s.SetOutputToWaveFile('{safe_path}'); "
            f"$s.Speak('{safe_text}'); "
            "$s.Dispose(); "
            "if ($v) { Write-Output $v.VoiceInfo.Name } else { Write-Output 'suara default' }"
        )

        result = self._run_powershell(script)
        if result is None:
            return None
        if result.returncode != 0 or not os.path.exists(self.wav_path):
            print(f"[WARNING] TTS gagal: {result.stderr.strip()[:200]}")
            return None

        name = result.stdout.strip()
        if name == "suara default":
            print(f"[WARNING] Tidak ada suara '{voice_prefix}' di SAPI5, "
                  "kalimat Indonesia akan dibaca dengan aksen suara default.")
        return name or "SAPI5"

    def trigger(self, label=""):
        """Bunyikan peringatan, dibatasi cooldown supaya tidak spam. -> True kalau jadi bunyi."""
        if self.muted:
            return False

        now = time.time()
        with self.lock:
            if now - self.last_played < self.cooldown:
                return False
            self.last_played = now

        print(f"[ALERT] {time.strftime('%H:%M:%S')} Orang tidak dikenal terdeteksi {label}".rstrip())
        threading.Thread(target=self._play, daemon=True).start()
        return True

    def _play(self):
        if winsound is None:
            return
        try:
            if self.ready:
                winsound.PlaySound(self.wav_path, winsound.SND_FILENAME | winsound.SND_ASYNC)
            else:
                for _ in range(3):
                    winsound.Beep(880, 220)
                    winsound.Beep(620, 220)
        except Exception as exc:                
            print(f"[WARNING] Gagal memutar peringatan: {exc}")

    def toggle_mute(self):
        self.muted = not self.muted
        print(f"[INFO] Suara peringatan {'OFF' if self.muted else 'ON'}")


def crop_face_thumbnail(img, bbox, size=PREVIEW_SIZE, margin=0.25):
    """Potong area wajah dari foto + kasih margin, lalu resize jadi kotak."""
    h, w = img.shape[:2]
    left, top, right, bottom = bbox.astype(int)

    pad_x = int((right - left) * margin)
    pad_y = int((bottom - top) * margin)

    left = max(0, left - pad_x)
    top = max(0, top - pad_y)
    right = min(w, right + pad_x)
    bottom = min(h, bottom + pad_y)

    return cv2.resize(img[top:bottom, left:right], (size, size))


def make_unknown_thumbnail(size=PREVIEW_SIZE):
    """Thumbnail placeholder untuk wajah yang tidak dikenali."""
    thumb = np.full((size, size, 3), 60, dtype=np.uint8)
    cv2.putText(thumb, "?", (int(size * 0.34), int(size * 0.68)),
                cv2.FONT_HERSHEY_DUPLEX, size / 55.0, (160, 160, 160), 2)
    return thumb


def person_name_from_filename(filename):
    """Buang nomor urut di belakang nama file supaya beberapa foto orang yang
    sama jadi satu identitas: 'William 3.jpg' / 'William_3.jpg' -> 'William'."""
    name = os.path.splitext(filename)[0]
    stripped = re.sub(r"[\s_\-\.]*\d+$", "", name).strip()
    return stripped if stripped else name


def collect_person_photos():
    """Kumpulkan daftar foto per orang. Dua cara pendaftaran didukung:

      known_faces/William/1.jpg, 2.jpg, ...        -> satu folder per orang
      known_faces/William 1.jpg, William 2.jpg     -> nama file + nomor urut
    """
    people = {}

    for entry in sorted(os.listdir(KNOWN_FACES_DIR)):
        path = os.path.join(KNOWN_FACES_DIR, entry)

        if os.path.isdir(path):
            photos = [os.path.join(path, f) for f in sorted(os.listdir(path))
                      if f.lower().endswith(IMAGE_EXTS)]
            if photos:
                people.setdefault(entry, []).extend(photos)
        elif entry.lower().endswith(IMAGE_EXTS):
            people.setdefault(person_name_from_filename(entry), []).append(path)

    return people


def biggest_face(faces):
    """Kalau di foto pendaftaran ada lebih dari satu wajah, ambil yang paling besar
    (paling dekat ke kamera) supaya tidak salah ambil orang di belakang."""
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))


def load_known_faces(face_app):
    """Load semua foto di known_faces/. Semua foto milik orang yang sama disimpan
    sebagai beberapa embedding untuk satu identitas, jadi variasi pose/cahaya/
    ekspresi ikut terdaftar dan pencocokan lebih akurat."""
    people = collect_person_photos()

    known_embeddings = []      
    embedding_owner = []       
    known_names = []           
    known_thumbs = []          
    known_counts = []          

    for name, photo_paths in sorted(people.items()):
        embeddings = []
        thumb = None

        for path in photo_paths:
            filename = os.path.basename(path)
            img = cv2.imread(path)

            if img is None:
                print(f"[WARNING] Gagal membaca file {filename}, dilewati.")
                continue

            faces = face_app.get(img)

            if len(faces) == 0:
                print(f"[WARNING] Tidak ada wajah terdeteksi di {filename}, dilewati.")
                continue

            face = biggest_face(faces)
            embeddings.append(face.normed_embedding)

            if thumb is None:
                thumb = crop_face_thumbnail(img, face.bbox)

        if not embeddings or thumb is None:
            print(f"[WARNING] {name} dilewati, tidak ada foto yang bisa dipakai.")
            continue

        person_idx = len(known_names)
        known_names.append(name)
        known_thumbs.append(thumb)
        known_counts.append(len(embeddings))

        known_embeddings.extend(embeddings)
        embedding_owner.extend([person_idx] * len(embeddings))

        catatan = "" if len(embeddings) >= PHOTOS_PER_PERSON else f" (disarankan {PHOTOS_PER_PERSON} foto)"
        print(f"[INFO] Loaded: {name} - {len(embeddings)}/{len(photo_paths)} foto dipakai{catatan}")

    return known_embeddings, embedding_owner, known_names, known_thumbs, known_counts


def cosine_similarity(a, b):
    return float(np.dot(a, b))


def iou(box_a, box_b):
    """Intersection over Union dua bounding box (x1, y1, x2, y2)."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h

    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


class Track:
    """Satu wajah yang sedang di-track, punya ID tetap selama masih kelihatan."""

    _next_id = 1

    def __init__(self, bbox, recog):
        self.id = Track._next_id
        Track._next_id += 1

        self.bbox = np.asarray(bbox, dtype=float)
        self.name_votes = deque(maxlen=VOTE_HISTORY)
        self.thumb_votes = deque(maxlen=VOTE_HISTORY)
        self.score = 0.0
        self.missed = 0
        self.alerted_at = 0.0

        self.frames_since_rec = self.id % max(REC_INTERVAL, 1)
        self.apply_recognition(recog)

    def apply_recognition(self, recog):
        name, score, thumb_idx = recog
        self.name_votes.append(name)
        self.thumb_votes.append(thumb_idx)
        self.score = score
        self.frames_since_rec = 0

    def due_for_recognition(self):
        return self.frames_since_rec >= REC_INTERVAL

    def update(self, bbox, recog=None):
        self.bbox = BOX_SMOOTHING * np.asarray(bbox, dtype=float) + (1 - BOX_SMOOTHING) * self.bbox
        self.missed = 0

        if recog is not None:
            self.apply_recognition(recog)
        else:
            self.frames_since_rec += 1

    def unknown_alert_due(self, now):
        """True kalau track ini sudah yakin Unknown dan belum (atau sudah lama)
        diperingatkan. Butuh beberapa kali recognition dulu supaya orang yang
        cuma sekali meleset dikenali tidak langsung membunyikan alarm."""
        if len(self.name_votes) < ALERT_VOTES or self.name != "Unknown":
            return False
        if self.alerted_at == 0.0:
            return True
        return ALERT_REPEAT > 0 and (now - self.alerted_at) >= ALERT_REPEAT

    @property
    def name(self):
        """Nama hasil voting dari beberapa recognition terakhir (anti kedap-kedip)."""
        return Counter(self.name_votes).most_common(1)[0][0]

    @property
    def thumb_idx(self):
        """Index orang yang paling sering cocok; -1 kalau Unknown."""
        if self.name == "Unknown":
            return -1
        return Counter([i for i in self.thumb_votes if i >= 0]).most_common(1)[0][0]

    @property
    def box_int(self):
        return self.bbox.astype(int)


def person_score(photo_scores):
    """Satu skor untuk satu orang dari skor semua foto pendaftarannya.
    TOP_K_MATCH = 1 berarti pakai foto yang paling mirip (paling toleran ke
    perbedaan pose); naikkan ke 2-3 kalau mulai ada salah kenal."""
    k = min(TOP_K_MATCH, len(photo_scores))
    if k <= 1:
        return float(np.max(photo_scores))
    return float(np.sort(photo_scores)[-k:].mean())


def recognize(frame, det, rec_model, known_matrix, person_rows, known_names):
    """Ambil embedding wajah lalu cocokkan ke semua foto known_faces, digabung
    per orang. -> (nama, score, person_idx)"""
    face = Face(bbox=det["bbox"], kps=det["kps"], det_score=det["det_score"])
    rec_model.get(frame, face)

    photo_scores = known_matrix @ face.normed_embedding
    scores = [person_score(photo_scores[rows]) for rows in person_rows]

    best_idx = int(np.argmax(scores))
    best_score = float(scores[best_idx])

    if best_score >= SIMILARITY_THRESHOLD:
        return known_names[best_idx], best_score, best_idx
    return "Unknown", best_score, -1


def match_tracks(tracks, detections):
    """Cocokkan track lama dengan deteksi baru pakai greedy IoU matching."""
    pairs = []
    for t_idx, track in enumerate(tracks):
        for d_idx, det in enumerate(detections):
            score = iou(track.bbox, det["bbox"])
            if score >= IOU_THRESHOLD:
                pairs.append((score, t_idx, d_idx))

    pairs.sort(reverse=True)

    used_tracks = set()
    used_dets = set()
    matches = []

    for _, t_idx, d_idx in pairs:
        if t_idx in used_tracks or d_idx in used_dets:
            continue
        used_tracks.add(t_idx)
        used_dets.add(d_idx)
        matches.append((t_idx, d_idx))

    unmatched_dets = [i for i in range(len(detections)) if i not in used_dets]
    unmatched_tracks = [i for i in range(len(tracks)) if i not in used_tracks]

    return matches, unmatched_dets, unmatched_tracks


def paste_thumbnail(frame, thumb, x, y, border_color=(255, 255, 255)):
    """Tempel thumbnail ke frame di posisi (x, y), otomatis dipotong kalau lewat tepi."""
    h, w = thumb.shape[:2]
    fh, fw = frame.shape[:2]

    x = max(0, min(x, fw - 1))
    y = max(0, min(y, fh - 1))

    crop_w = min(w, fw - x)
    crop_h = min(h, fh - y)

    if crop_w <= 0 or crop_h <= 0:
        return

    frame[y:y + crop_h, x:x + crop_w] = thumb[:crop_h, :crop_w]
    cv2.rectangle(frame, (x, y), (x + crop_w - 1, y + crop_h - 1), border_color, 2)


def draw_side_panel(frame, tracks, known_counts, known_thumbs, unknown_thumb):
    """Panel di kiri: daftar orang yang lagi kelihatan + foto referensinya."""
    if not tracks:
        return

    pad = 10
    row_h = PREVIEW_SIZE + pad
    panel_w = PREVIEW_SIZE + 190
    panel_h = pad + row_h * len(tracks)
    panel_h = min(panel_h, frame.shape[0] - 2 * pad)

    overlay = frame.copy()
    cv2.rectangle(overlay, (pad, pad), (pad + panel_w, pad + panel_h), (25, 25, 25), cv2.FILLED)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    y = pad + pad // 2
    for track in tracks:
        if y + PREVIEW_SIZE > pad + panel_h:
            break

        idx = track.thumb_idx
        thumb = known_thumbs[idx] if idx >= 0 else unknown_thumb
        known = track.name != "Unknown"
        color = (0, 255, 0) if known else (0, 0, 255)

        paste_thumbnail(frame, thumb, pad + pad // 2, y, color)

        text_x = pad + pad // 2 + PREVIEW_SIZE + 12
        cv2.putText(frame, f"#{track.id} {track.name}", (text_x, y + 30),
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, color, 1)
        cv2.putText(frame, f"score: {track.score:.2f}", (text_x, y + 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        ref_text = f"ref: {known_counts[idx]} foto" if idx >= 0 else "ref: -"
        cv2.putText(frame, ref_text, (text_x, y + 82),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (170, 170, 170), 1)

        y += row_h


def set_fullscreen(window, enable):
    """Ubah jendela preview jadi fullscreen atau kembali ke mode normal."""
    cv2.setWindowProperty(
        window, cv2.WND_PROP_FULLSCREEN,
        cv2.WINDOW_FULLSCREEN if enable else cv2.WINDOW_NORMAL)


def main():
    print("[INFO] Loading model InsightFace (pertama kali akan download model, mohon tunggu)...")
    face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"],
                            allowed_modules=["detection", "recognition"])
    face_app.prepare(ctx_id=0, det_size=(DET_SIZE, DET_SIZE))

    det_model = face_app.models["detection"]
    rec_model = face_app.models["recognition"]

    known_embeddings, embedding_owner, known_names, known_thumbs, known_counts = load_known_faces(face_app)

    if len(known_embeddings) == 0:
        print("[ERROR] Tidak ada wajah yang berhasil di-load. Cek folder known_faces/")
        return

    unknown_thumb = make_unknown_thumbnail()
    known_matrix = np.vstack(known_embeddings)

    alert = IntruderAlert() if ALERT_ENABLED else None

    owner = np.asarray(embedding_owner)
    person_rows = [np.where(owner == i)[0] for i in range(len(known_names))]

    print(f"[INFO] Terdaftar {len(known_names)} orang dari {len(known_embeddings)} foto.")

    print(f"[INFO] Menghubungkan ke CCTV: rtsp://{CAM_USERNAME}:****@{CAM_IP}:{CAM_PORT}/Streaming/Channels/{CAM_CHANNEL}")

    sources = [RTSP_URL]
    if WEBCAM_FALLBACK:
        sources.append(WEBCAM_INDEX)
        print(f"[INFO] Fallback ke webcam #{WEBCAM_INDEX} kalau RTSP tidak bisa dibuka.")
    camera = CameraStream(sources)

    if not camera.is_opened():
        print("[ERROR] Tidak bisa membuka stream RTSP maupun webcam. Cek IP/kredensial CCTV atau webcam yang terpasang.")
        return

    if not camera.start():
        print("[ERROR] Stream RTSP terbuka tapi tidak mengirim frame (timeout).")
        camera.stop()
        return

    print("[INFO] CCTV aktif. Tekan 'q' keluar, 'p' toggle panel, 's' matikan/nyalakan suara peringatan.")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    fullscreen = START_FULLSCREEN
    set_fullscreen(WINDOW_NAME, fullscreen)

    tracks = []
    show_panel = SHOW_SIDE_PANEL
    fps = 0.0
    prev_time = time.time()

    while True:
        frame = camera.read()
        if frame is None:
            time.sleep(0.05)
            continue

        bboxes, kpss = det_model.detect(frame, max_num=0, metric="default")

        detections = [
            {"bbox": bboxes[i][:4].astype(float), "kps": kpss[i], "det_score": float(bboxes[i][4])}
            for i in range(len(bboxes))
        ]

        matches, unmatched_dets, unmatched_tracks = match_tracks(tracks, detections)

        for t_idx, d_idx in matches:
            track = tracks[t_idx]
            det = detections[d_idx]

            recog = None
            if track.due_for_recognition():
                recog = recognize(frame, det, rec_model, known_matrix, person_rows, known_names)

            track.update(det["bbox"], recog)

        for t_idx in unmatched_tracks:
            tracks[t_idx].missed += 1

        for d_idx in unmatched_dets:
            det = detections[d_idx]
            recog = recognize(frame, det, rec_model, known_matrix, person_rows, known_names)
            tracks.append(Track(det["bbox"], recog))

        tracks = [t for t in tracks if t.missed <= MAX_MISSED_FRAMES]
        visible_tracks = [t for t in tracks if t.missed == 0]

        if alert is not None:
            now_ts = time.time()
            for track in visible_tracks:
                if track.unknown_alert_due(now_ts) and alert.trigger(f"(track #{track.id})"):
                    track.alerted_at = now_ts

        for track in visible_tracks:
            left, top, right, bottom = track.box_int
            known = track.name != "Unknown"
            color = (0, 255, 0) if known else (0, 0, 255)
            label = f"#{track.id} {track.name} ({track.score:.2f})"

            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(frame, (left, bottom), (right, bottom + 25), color, cv2.FILLED)
            cv2.putText(frame, label, (left + 6, bottom + 18),
                        cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

            idx = track.thumb_idx
            thumb = known_thumbs[idx] if idx >= 0 else unknown_thumb
            thumb_y = top - PREVIEW_SIZE - 6
            if thumb_y < 0:
                thumb_y = bottom + 30
            paste_thumbnail(frame, thumb, left, thumb_y, color)

        if show_panel:
            draw_side_panel(frame, visible_tracks, known_counts, known_thumbs, unknown_thumb)

        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 1e-6))
        prev_time = now
        cv2.putText(frame, f"FPS: {fps:.1f} | tracks: {len(visible_tracks)}",
                    (30, frame.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)

        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("f"):
            fullscreen = not fullscreen
            set_fullscreen(WINDOW_NAME, fullscreen)
        if key == 27 and fullscreen:
            fullscreen = False
            set_fullscreen(WINDOW_NAME, fullscreen)
        if key == ord("p"):
            show_panel = not show_panel
        if key == ord("s") and alert is not None:
            alert.toggle_mute()

    camera.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()