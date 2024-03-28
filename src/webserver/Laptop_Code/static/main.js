document.addEventListener('DOMContentLoaded', function () {
    let video = document.getElementById('video');
    let recordBtn = document.getElementById('recordBtn');
    let statusText = document.getElementById('status');
    let mediaRecorder;
    let videoChunks = [];

    if (navigator.mediaDevices.getUserMedia) {
        navigator.mediaDevices.getUserMedia({ video: true })
            .then(function (stream) {
                video.srcObject = stream;
                mediaRecorder = new MediaRecorder(stream);

                mediaRecorder.ondataavailable = function(e) {
                    videoChunks.push(e.data);
                };

                mediaRecorder.onstop = function() {
                    let blob = new Blob(videoChunks, { 'type' : 'video/mp4' });
                    videoChunks = [];
                    sendData(blob);
                    statusText.innerText = "Recording finished. Processing...";
                };
            })
            .catch(function (err) {
                console.log("Issue with accessing webcam", err);
            });
    }

    recordBtn.addEventListener('click', function() {
        if (mediaRecorder.state === "inactive") {
            mediaRecorder.start();
            statusText.innerText = "Recording in progress...";
            setTimeout(() => {
                if (mediaRecorder.state === "recording") {
                    mediaRecorder.stop();
                }
            }, 13500);
        }
    });
});

function sendData(videoBlob) {
    let formData = new FormData();
    formData.append('video', videoBlob, 'video.mp4');

    fetch('/process', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        console.log(data);
        document.getElementById('status').innerText = data.message;
    })
    .catch((error) => {
        console.error('Error:', error);
        document.getElementById('status').innerText = "Error sending video.";
    });
}
