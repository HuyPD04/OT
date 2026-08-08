FROM python:3.11-slim-bookworm AS builder

WORKDIR /build

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --prefix=/install --no-cache-dir -r requirements.txt

FROM python:3.11-slim-bookworm

WORKDIR /app

EXPOSE 8080

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONPATH=/usr/lib/python3/dist-packages

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-gi \
    python3-gi-cairo \
    gir1.2-gstreamer-1.0 \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-libav \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN gst-inspect-1.0 rtspsrc >/dev/null \
    && gst-inspect-1.0 rtph264depay >/dev/null \
    && gst-inspect-1.0 rtph265depay >/dev/null \
    && gst-inspect-1.0 h264parse >/dev/null \
    && gst-inspect-1.0 h265parse >/dev/null \
    && gst-inspect-1.0 avdec_h264 >/dev/null \
    && gst-inspect-1.0 avdec_h265 >/dev/null

COPY --from=builder /install /usr/local

COPY src ./src

CMD ["python", "-m", "src.main"]
