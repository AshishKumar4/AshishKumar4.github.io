const video = document.getElementById('webcam');
const predView = document.getElementById('prediction');
const frameDisplay = document.getElementById('frames')

const liveView = document.getElementById('liveView');
const demosSection = document.getElementById('demos');
const enableWebcamButton = document.getElementById('webcamButton');

// Check if webcam access is supported.
function getUserMediaSupported() {
    return !!(navigator.mediaDevices &&
        navigator.mediaDevices.getUserMedia);
}

// If webcam supported, add event listener to button for when user
// wants to activate it to call enableCam function which we will 
// define in the next step.
if (getUserMediaSupported()) {
    enableWebcamButton.addEventListener('click', enableCam);
} else {
    console.warn('getUserMedia() is not supported by your browser');
}

async function toPixels(tensor, canvas=null){
    const [height, width] = tensor.shape.slice(0, 2);
    // convert to rgba by adding alpha channel
    const alpha = tf.fill([height, width, 1], 255, 'int32');
    const rgba = tf.concat([tensor, alpha], 2);
    tf.dispose([alpha, tensor]);
    
    const bytes = await rgba.data();
    const pixelData = new Uint8ClampedArray(bytes);
    if (canvas !== null){
        const imageData = new ImageData(pixelData, width, height);
        const ctx = canvas.getContext('2d');
        ctx.putImageData(imageData, 0, 0);
    }
    tf.dispose(rgba);
    return bytes;
}

// Enable the live webcam view and start classification.
async function enableCam(event) {
    // // Hide the button once clicked.
    event.target.classList.add('removed');

    // // getUsermedia parameters to force video but not audio.
    const constraints = {
        video: true
    };

    // // Activate the webcam stream.
    navigator.mediaDevices.getUserMedia(constraints).then(function (stream) {
        // video.srcObject = stream;
        video.addEventListener('loadeddata', predictWebcam);
    });
    webcam = await tf.data.webcam(video);
}

var model = undefined;
var modelInputShape = undefined;
var webcam = undefined;
var predictions = undefined;
var frames = 0;

function timer() {
    frameDisplay.innerHTML = frames.toString()
    frames = 0;
}

async function predictSegmentation(img, raw) {
    // Make a prediction through our newly-trained model using the embeddings
    // from mobilenet as input.
    await tf.tidy(() => {
        predictions = model.predict(img).squeeze().softmax();
        let [background, person] = predictions.resizeBilinear([480, 640]).split(2, 2);
        // pmin = person.min();
        // pmax = person.max();
        // person = person.sub(pmin).div(pmax.sub(pmin)).sub(0.5).ceil()
        person = person.mul(raw.squeeze());
        if (frames % 2 == 0) {
            // person = person.resizeNearestNeighbor([96, 160]);
            toPixels(person.mul(255).asType('int32'), predView);
        }

        background.dispose()
        person.dispose()
    })
}

async function predictWebcam() {
    console.log("Here");
    while (true) {
        // Capture the frame from the webcam.
        const [raw, img] = await getImage();
        // Only Render on alternate frames
        await predictSegmentation(img, raw)
        img.dispose()
        raw.dispose()
        frames += 1;
        await tf.nextFrame();
    }
}

/**
 * Captures a frame from the webcam and normalizes it between -1 and 1.
 * Returns a batched image (1-element batch) of shape [1, w, h, c].
 */
async function getImage() {
    const img = await webcam.capture();
    const rawProcessed = tf.tidy(() => img.div(255).expandDims(0).toFloat());
    const finalProcessed = tf.tidy(() => rawProcessed.resizeBilinear(modelInputShape));
    img.dispose();
    return [rawProcessed, finalProcessed];
}

async function init() {
    // Store the resulting model in the global scope of our app.
    model = await tf.loadGraphModel('model.json');
    modelInputShape = model.inputs[0].shape;
    modelInputShape = [modelInputShape[1], modelInputShape[2]]
    demosSection.classList.remove('invisible');
    setInterval(timer, 1000);
}

init();

// tf.setBackend('webgl').then(() => init());
// tf.setBackend('wasm').then(() => init());
