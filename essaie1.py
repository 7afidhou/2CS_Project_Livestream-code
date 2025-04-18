from flask import Flask, render_template, Response
import subprocess

app = Flask(__name__)

# Function to capture video frames using libcamera-vid
def gen_frames():
    # Run libcamera-vid in MJPEG format and stream the output
    command = [
        "libcamera-vid",
        "-t", "0",  # Duration set to 0 for infinite (streaming)
        "--width", "640",
        "--height", "480",
        "--codec", "mjpeg",
        "-o", "-",  # Output to stdout
        "--nopreview"
    ]
    
    process = subprocess.Popen(command, stdout=subprocess.PIPE)
    
    buffer = b""
    while True:
        buffer += process.stdout.read(4096)
        
        # Look for JPEG frame markers
        jpg_start = buffer.find(b"\xff\xd8")
        jpg_end = buffer.find(b"\xff\xd9")

        if jpg_start != -1 and jpg_end != -1:
            frame_data = buffer[jpg_start:jpg_end+2]
            buffer = buffer[jpg_end+2:]
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')

@app.route('/')
def index():
    return render_template('indexx.html')  # Ensure index.html exists

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
