# Integrated Cihampelas Mobility AI Backend

Prototype backend:
video upload -> YOLO detection -> ByteTrack tracking -> counting -> traffic metrics -> JSON result.

Target classes:
- motorcycle -> motor
- car -> mobil (termasuk angkot/taxi/travel)
- bus -> bus
- truck -> truk
- bicycle -> sepeda

Endpoint:
POST /analyze
GET /health
GET /docs
GET /demo

`/analyze` menerima multipart:
- video
- location
- config (JSON)

Kecepatan m/s hanya dihitung jika speed_line_a, speed_line_b, dan speed_distance_m tersedia.

Video asli disimpan dan dikembalikan sebagai `source_video_url`; backend tidak merender video baru untuk kebutuhan LED. Metadata deteksi dikembalikan sebagai timeline agar frontend dapat meng-overlay hasil AI di atas video asli.

Queue detection belum diaktifkan pada baseline ini untuk menghindari angka palsu.

Deployment:
Gunakan Dockerfile + Railway. Setelah deploy, buat public domain lalu endpoint menjadi:
https://DOMAIN/analyze

Periksa lisensi Ultralytics sebelum penggunaan publik/komersial.
