from flask import Flask, render_template, request
import os
import socket
import secrets
import string
from typing import List
import io


TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")
app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=None)
app.config["TEMPLATES_AUTO_RELOAD"] = True


@app.after_request
def add_no_cache_headers(response):
    # Encourage the browser to always fetch fresh HTML/CSS/JS during development
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.route("/debug-template")
def debug_template():
    base_path = os.path.join(TEMPLATE_DIR, "base.html")
    try:
        mtime = os.path.getmtime(base_path)
    except Exception:
        mtime = None
    return {
        "template_dir": TEMPLATE_DIR,
        "base_exists": os.path.exists(base_path),
        "base_mtime": mtime,
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/password", methods=["GET", "POST"])
def password_generator():
    generated_passwords: List[str] = []
    error_message: str = ""

    if request.method == "POST":
        try:
            length = int(request.form.get("length", 16))
        except ValueError:
            length = 16

        use_upper = request.form.get("upper") == "on"
        use_lower = request.form.get("lower") == "on"
        use_digits = request.form.get("digits") == "on"
        use_symbols = request.form.get("symbols") == "on"
        avoid_ambiguous = request.form.get("avoid_ambiguous") == "on"
        count = min(max(int(request.form.get("count", 1)), 1), 10)

        if not any([use_upper, use_lower, use_digits, use_symbols]):
            error_message = "Select at least one character set."
        else:
            ambiguous_chars = "Il1O0|`'\"{}[]()/\\,.;:" if avoid_ambiguous else ""

            pool = ""
            if use_upper:
                pool += string.ascii_uppercase
            if use_lower:
                pool += string.ascii_lowercase
            if use_digits:
                pool += string.digits
            if use_symbols:
                pool += "!@#$%^&*_-+=?"

            if ambiguous_chars:
                pool = "".join(ch for ch in pool if ch not in ambiguous_chars)

            if not pool:
                error_message = "Character pool is empty after filters."
            else:
                selected_sets = []
                if use_upper:
                    selected_sets.append([c for c in string.ascii_uppercase if c in pool])
                if use_lower:
                    selected_sets.append([c for c in string.ascii_lowercase if c in pool])
                if use_digits:
                    selected_sets.append([c for c in string.digits if c in pool])
                if use_symbols:
                    selected_sets.append([c for c in "!@#$%^&*_-+=?" if c in pool])

                for _ in range(count):
                    password_chars: List[str] = []
                    for charset in selected_sets:
                        if charset:
                            password_chars.append(secrets.choice(charset))
                    while len(password_chars) < length:
                        password_chars.append(secrets.choice(pool))
                    secrets.SystemRandom().shuffle(password_chars)
                    generated_passwords.append("".join(password_chars[:length]))

    return render_template(
        "password.html",
        passwords=generated_passwords,
        error_message=error_message,
    )

@app.route("/qr", methods=["GET", "POST"])
def qr_scanner():
    decoded_results: List[str] = []
    error_message: str = ""

    if request.method == "POST":
        file = request.files.get("qr_image")
        if not file or file.filename == "":
            error_message = "Please upload an image containing a QR code."
        else:
            try:
                import numpy as np  # type: ignore
                import cv2  # type: ignore

                image_bytes = file.read()
                if not image_bytes:
                    raise ValueError("Empty file uploaded")

                image_array = np.frombuffer(image_bytes, dtype=np.uint8)
                image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError("Unsupported or corrupted image file")

                detector = cv2.QRCodeDetector()
                retval, decoded_infos, points, _ = detector.detectAndDecodeMulti(image)
                if retval and decoded_infos:
                    decoded_results = [s for s in decoded_infos if s]
                else:
                    data, _, _ = detector.detectAndDecode(image)
                    if data:
                        decoded_results = [data]

                if not decoded_results:
                    error_message = "No QR code detected. Try a clearer image."

            except Exception as exc:  # noqa: BLE001
                error_message = f"Failed to process image: {exc}"

    return render_template(
        "qr.html",
        results=decoded_results,
        error_message=error_message,
    )

@app.route("/exif", methods=["GET", "POST"])
def exif_viewer():
    from typing import Dict

    exif_data: Dict[str, str] = {}
    error_message: str = ""

    if request.method == "POST":
        file = request.files.get("photo")
        if not file or file.filename == "":
            error_message = "Please upload a photo (JPEG/PNG)."
        else:
            try:
                from PIL import Image, ExifTags  # type: ignore

                img_bytes = io.BytesIO(file.read())
                img = Image.open(img_bytes)
                exif_raw = getattr(img, "_getexif", lambda: None)()
                if exif_raw:
                    # Build tag maps
                    tag_map = {k: ExifTags.TAGS.get(k, str(k)) for k in exif_raw.keys()}
                    # Extract GPS if present
                    gps_info = None
                    for tag_id, tag_name in tag_map.items():
                        if tag_name == "GPSInfo":
                            gps_info = exif_raw.get(tag_id)
                            break

                    def _to_float(rational):
                        try:
                            # Pillow may give (num, den) or a Fraction-like
                            if hasattr(rational, 'numerator') and hasattr(rational, 'denominator'):
                                return float(rational.numerator) / float(rational.denominator or 1)
                            if isinstance(rational, tuple) and len(rational) == 2:
                                num, den = rational
                                return float(num) / float(den or 1)
                            return float(rational)
                        except Exception:
                            return None

                    def _convert_gps_to_deg(gps):
                        try:
                            from PIL.ExifTags import GPSTAGS  # type: ignore
                            gps_parsed = {GPSTAGS.get(k, k): v for k, v in gps.items()}
                            lat_vals = gps_parsed.get('GPSLatitude')
                            lat_ref = gps_parsed.get('GPSLatitudeRef')
                            lon_vals = gps_parsed.get('GPSLongitude')
                            lon_ref = gps_parsed.get('GPSLongitudeRef')
                            alt_val = gps_parsed.get('GPSAltitude')

                            def dms_to_deg(dms):
                                d = _to_float(dms[0])
                                m = _to_float(dms[1])
                                s = _to_float(dms[2])
                                if None in (d, m, s):
                                    return None
                                return d + (m / 60.0) + (s / 3600.0)

                            lat = dms_to_deg(lat_vals) if lat_vals else None
                            lon = dms_to_deg(lon_vals) if lon_vals else None
                            if lat is not None and lat_ref in ['S', 's']:
                                lat = -lat
                            if lon is not None and lon_ref in ['W', 'w']:
                                lon = -lon
                            alt = _to_float(alt_val) if alt_val is not None else None

                            return lat, lon, alt
                        except Exception:
                            return None, None, None

                    # Select only OSINT-relevant fields
                    wanted_tags = [
                        "DateTimeOriginal", "DateTime", "CreateDate", "Make", "Model",
                        "LensModel", "Software", "FNumber", "ExposureTime", "ISOSpeedRatings",
                        "PhotographicSensitivity", "FocalLength",
                    ]

                    # Populate selected fields if present
                    for tag_id, value in exif_raw.items():
                        tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
                        if tag_name in wanted_tags:
                            try:
                                exif_data[tag_name] = str(value)
                            except Exception:
                                exif_data[tag_name] = "<unreadable>"

                    # GPS conversion
                    if gps_info:
                        lat, lon, alt = _convert_gps_to_deg(gps_info)
                        if lat is not None and lon is not None:
                            exif_data["GPSLatitude"] = f"{lat:.6f}"
                            exif_data["GPSLongitude"] = f"{lon:.6f}"
                            exif_data["GPSMapLink"] = f"https://maps.google.com/?q={lat:.6f},{lon:.6f}"
                        if alt is not None:
                            exif_data["GPSAltitude"] = f"{alt:.1f} m"
                else:
                    error_message = "No EXIF metadata found in this image."
            except Exception as exc:  # noqa: BLE001
                error_message = f"Failed to read image: {exc}"

    return render_template(
        "exif.html",
        exif=exif_data,
        error_message=error_message,
    )


@app.route("/whois", methods=["GET", "POST"])
def whois_lookup():
    """WHOIS + IP info lookup for a domain or host."""
    from typing import Any, Dict
    import ipaddress
    results: Dict[str, Any] = {"whois": None, "ipinfo": None}
    error_message: str = ""

    if request.method == "POST":
        target = (request.form.get("domain") or "").strip()
        if not target:
            error_message = "Please enter a domain or IP address."
        else:
            # Resolve to IP (if domain)
            ip_address_str = None
            try:
                # If it's already an IP, validate it; else resolve
                try:
                    ipaddress.ip_address(target)
                    ip_address_str = target
                except ValueError:
                    ip_address_str = socket.gethostbyname(target)
            except Exception as exc:  # noqa: BLE001
                error_message = f"Host resolution failed: {exc}"

            # WHOIS lookup (domain only; skip for direct IPs typically)
            try:
                import whois  # type: ignore
                domain_for_whois = target
                # If user input was an IP, WHOIS may be less informative
                w = whois.whois(domain_for_whois)
                # Convert complex types to strings
                clean = {}
                for k, v in (w or {}).items():
                    try:
                        if isinstance(v, (list, tuple)):
                            clean[k] = ", ".join(str(x) for x in v if x)
                        else:
                            clean[k] = str(v)
                    except Exception:
                        clean[k] = "<unreadable>"
                results["whois"] = clean
            except Exception as exc:  # noqa: BLE001
                # Non-fatal; continue with IP info
                results["whois"] = {"note": f"WHOIS lookup failed: {exc}"}

            # IP geo/info via ip-api.com (no key, public) if we have an IP
            if ip_address_str:
                try:
                    import requests  # type: ignore
                    resp = requests.get(f"http://ip-api.com/json/{ip_address_str}", timeout=5)
                    if resp.ok:
                        results["ipinfo"] = resp.json()
                    else:
                        results["ipinfo"] = {"status": "fail", "message": f"HTTP {resp.status_code}"}
                except Exception as exc:  # noqa: BLE001
                    results["ipinfo"] = {"status": "fail", "message": str(exc)}

    return render_template("whois.html", results=results, error_message=error_message)


@app.route("/stego", methods=["GET", "POST"])
def steganography_tool():
    """Simple LSB steganography embed/extract for educational purposes."""
    from typing import Optional, Tuple
    from PIL import Image  # type: ignore

    error_message: str = ""
    success_message: str = ""
    downloaded_filename: Optional[str] = None
    output_image_bytes: Optional[bytes] = None
    output_image_b64: Optional[str] = None
    extracted_text: Optional[str] = None

    def _bytes_to_bits(data: bytes) -> list[int]:
        bits: list[int] = []
        for byte in data:
            for i in range(7, -1, -1):
                bits.append((byte >> i) & 1)
        return bits

    def _bits_to_bytes(bits: list[int]) -> bytes:
        if len(bits) % 8 != 0:
            bits = bits[: len(bits) - (len(bits) % 8)]
        data = bytearray()
        for i in range(0, len(bits), 8):
            b = 0
            for j in range(8):
                b = (b << 1) | (bits[i + j] & 1)
            data.append(b)
        return bytes(data)

    def _capacity_pixels(width: int, height: int) -> int:
        # Each pixel (RGB) can carry 3 bits (one per channel)
        return width * height * 3

    def _encode_lsb(base_img: Image.Image, message: str) -> Tuple[bytes, str]:
        # Prepare payload: 32-bit big-endian length + utf-8 bytes
        payload = message.encode("utf-8")
        length_bits = len(payload) * 8
        header = length_bits.to_bytes(4, byteorder="big")
        all_bytes = header + payload
        bits = _bytes_to_bits(all_bytes)

        img = base_img.convert("RGB")
        width, height = img.size
        capacity = _capacity_pixels(width, height)
        if len(bits) > capacity:
            raise ValueError(
                f"Message too large. Need {len(bits)} bits but image holds {capacity} bits."
            )

        pixels = list(img.getdata())
        bit_index = 0
        new_pixels = []
        for r, g, b in pixels:
            if bit_index < len(bits):
                r = (r & ~1) | bits[bit_index]
                bit_index += 1
            if bit_index < len(bits):
                g = (g & ~1) | bits[bit_index]
                bit_index += 1
            if bit_index < len(bits):
                b = (b & ~1) | bits[bit_index]
                bit_index += 1
            new_pixels.append((r, g, b))

        img.putdata(new_pixels)
        out_io = io.BytesIO()
        img.save(out_io, format="PNG")
        return out_io.getvalue(), "Embedded text into image (PNG)."

    def _decode_lsb(base_img: Image.Image) -> str:
        img = base_img.convert("RGB")
        pixels = list(img.getdata())
        bits: list[int] = []
        for r, g, b in pixels:
            bits.extend([r & 1, g & 1, b & 1])

        # First 32 bits = payload length (in bits)
        header_bits = bits[:32]
        header_bytes = _bits_to_bytes(header_bits)
        if len(header_bytes) < 4:
            raise ValueError("Image does not contain a valid header")
        payload_len_bits = int.from_bytes(header_bytes[:4], byteorder="big")
        if payload_len_bits <= 0:
            return ""
        payload_bits = bits[32 : 32 + payload_len_bits]
        payload_bytes = _bits_to_bytes(payload_bits)
        try:
            return payload_bytes.decode("utf-8", errors="strict")
        except Exception:
            # Fallback to replace errors to show something
            return payload_bytes.decode("utf-8", errors="replace")

    if request.method == "POST":
        mode = request.form.get("mode")
        if mode == "embed":
            src = request.files.get("cover_image")
            text = request.form.get("secret_text", "")
            if not src or src.filename == "":
                error_message = "Please upload a cover image."
            elif not text:
                error_message = "Please enter text to embed."
            else:
                try:
                    import base64
                    img = Image.open(io.BytesIO(src.read()))
                    output_image_bytes, success_message = _encode_lsb(img, text)
                    downloaded_filename = "stego_output.png"
                    output_image_b64 = base64.b64encode(output_image_bytes).decode("ascii")
                except Exception as exc:  # noqa: BLE001
                    error_message = f"Embedding failed: {exc}"
        elif mode == "extract":
            stego = request.files.get("stego_image")
            if not stego or stego.filename == "":
                error_message = "Please upload an image to extract from."
            else:
                try:
                    img = Image.open(io.BytesIO(stego.read()))
                    extracted_text = _decode_lsb(img)
                    if not extracted_text:
                        success_message = "No hidden text detected or message is empty."
                except Exception as exc:  # noqa: BLE001
                    error_message = f"Extraction failed: {exc}"

    return render_template(
        "stego.html",
        error_message=error_message,
        success_message=success_message,
        output_image_bytes=output_image_bytes,
        output_image_b64=output_image_b64,
        downloaded_filename=downloaded_filename,
        extracted_text=extracted_text,
    )

@app.route("/port-scanner", methods=["GET", "POST"])
def port_scanner():
    import socket
    from typing import Dict, List as TList

    scan_results: TList[Dict[str, str]] = []
    error_message: str = ""

    # Default common ports map
    default_ports = [80, 443, 22, 21, 25, 110, 143, 587, 3306, 3389, 8080]

    if request.method == "POST":
        target_host = (request.form.get("host") or "").strip()
        ports_input = (request.form.get("ports") or "").strip()
        timeout_s = float(request.form.get("timeout", 0.5))

        if not target_host:
            error_message = "Please enter a host or IP address."
        else:
            # Build port list
            ports: TList[int] = []
            if ports_input:
                for token in ports_input.split(','):
                    token = token.strip()
                    if not token:
                        continue
                    # support ranges like 20-25
                    if '-' in token:
                        try:
                            start_str, end_str = token.split('-', 1)
                            start, end = int(start_str), int(end_str)
                            if start <= end and 1 <= start <= 65535 and 1 <= end <= 65535:
                                ports.extend(list(range(start, end + 1)))
                        except Exception:
                            pass
                    else:
                        try:
                            p = int(token)
                            if 1 <= p <= 65535:
                                ports.append(p)
                        except ValueError:
                            pass
            if not ports:
                ports = default_ports

            # Safety cap
            ports = ports[:50]

            # Resolve once
            try:
                target_ip = socket.gethostbyname(target_host)
            except Exception as exc:  # noqa: BLE001
                error_message = f"Host resolution failed: {exc}"
                target_ip = None

            if target_ip:
                for port in ports:
                    status = "closed"
                    try:
                        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                            sock.settimeout(timeout_s)
                            result = sock.connect_ex((target_ip, port))
                            if result == 0:
                                status = "open"
                    except Exception:
                        status = "error"
                    scan_results.append({"port": str(port), "status": status})

    return render_template(
        "port.html",
        results=scan_results,
        error_message=error_message,
    )

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)


