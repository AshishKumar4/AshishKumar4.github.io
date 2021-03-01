const video = document.getElementById('webcam');
const predView = document.getElementById('prediction');
const frameDisplay = document.getElementById('frames');

const liveView = document.getElementById('liveView');
const demosSection = document.getElementById('demos');
const enableWebcamButton = document.getElementById('webcamButton');

const bgCanvas = document.getElementById('bgCanvas');
const bg_ctx = bgCanvas.getContext("2d");

const smoothFilter5 = tf.tensor4d([
    [[[1]], [[1]], [[1]], [[1]], [[1]]],
    [[[1]], [[1]], [[1]], [[1]], [[1]]],
    [[[1]], [[1]], [[1]], [[1]], [[1]]],
    [[[1]], [[1]], [[1]], [[1]], [[1]]],
    [[[1]], [[1]], [[1]], [[1]], [[1]]],
]).div(25);

const smoothFilter3 = tf.tensor4d([
    [[[1]], [[1]], [[1]]],
    [[[1]], [[1]], [[1]]],
    [[[1]], [[1]], [[1]]],
]).div(9);

var backgroundImage = undefined;

var model = undefined;
var modelInputShape = undefined;
var streamShape = undefined;
var webcam = undefined;
var predictions = undefined;
var frames = 0;

function readImage() {
    if (!this.files || !this.files[0]) return;

    const FR = new FileReader();
    FR.addEventListener("load", (evt) => {
        const img = new Image();
        img.style.objectFit = "fill";
        img.addEventListener("load", () => {
            backgroundImage = img;
            // bg_ctx.clearRect(0, 0, bg_ctx.canvas.width, bg_ctx.canvas.height);
            // bg_ctx.drawImage(img, 0, 0);
            // backgroundImage = tf.tidy(() => tf.browser.fromPixels(bgCanvas).asType('float32').div(255));
            // const ctx = predView.getContext('2d')
            // ctx.drawImage(bgCanvas, 0, 0)
            // ctx.save()
        });
        img.src = evt.target.result;
    });
    FR.readAsDataURL(this.files[0]);
}

document.getElementById('bgUpload').addEventListener("change", readImage);

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
		const { height, width } = stream.getVideoTracks()[0].getSettings();
		streamShape = [height, width];
        video.addEventListener('loadeddata', predictWebcam);
    });
    webcam = await tf.data.webcam(video);
}

function timer() {
    frameDisplay.innerHTML = frames.toString();
    frames = 0;
}

async function toPixels(tensor, canvas = null) {
    const [height, width] = tensor.shape.slice(0, 2);
    // convert to rgba by adding alpha channel
    // let alpha = undefined;
    // if (mask != undefined)
    // {
    //     alpha = mask.mul(255).asType('int32')
    // }
    // else 
    // {
    const alpha = tf.fill([height, width, 1], 255, 'int32');
    // }
    const rgba = tf.concat([tensor, alpha], 2);
    tf.dispose([alpha, tensor]);

    const bytes = await rgba.data();
    const pixelData = new Uint8ClampedArray(bytes);
    if (canvas !== null) {
        const imageData = new ImageData(pixelData, width, height);
        const ctx = canvas.getContext('2d');
        // ctx.restore()
        ctx.putImageData(imageData, 0, 0);
        // await createImageBitmap(imageData).then(function (imgBitmap) {
        //     ctx.drawImage(imgBitmap, 0, 0);
        // });
    }
    tf.dispose(rgba);
    return bytes;
}

async function predictSegmentation(img) {
    const [height, width] = streamShape;
    const [modelHeight, modelWidth] = modelInputShape;

    const imageData = new ImageData(modelWidth, modelHeight);
    // let imageData;

    tf.tidy(() => {
        const predictions = model.predict(img).squeeze().softmax().log().add(1).clipByValue(0, 1); // .resizeBilinear(streamShape); // .softmax().log().add(1).clipByValue(0, 1)
        const [background, person] = predictions.split(2, 2);
        const segmentationMask = background;

        const segData = segmentationMask.dataSync();
        for (let i = 0; i < modelHeight * modelWidth; i += 1) {
            imageData.data[i * 4 + 3] = segData[i] * 255;
        }
        segmentationMask.dispose();
        background.dispose();
        person.dispose();
    });

    const imageBitmap = await createImageBitmap(imageData);
    predViewCtx = predView.getContext('2d');
    predViewCtx.globalCompositeOperation = 'copy';
    predViewCtx.filter = 'blur(2px)';
    predViewCtx.drawImage(imageBitmap, 0, 0, modelWidth, modelHeight, 0, 0, width, height);
    predViewCtx.globalCompositeOperation = 'source-out';
    predViewCtx.filter = 'none';
    predViewCtx.drawImage(video, 0, 0);

    if (backgroundImage !== undefined && backgroundImage !== null) {
        predViewCtx.filter = 'none';
        predViewCtx.globalCompositeOperation = 'destination-over';
        predViewCtx.drawImage(backgroundImage, 0, 0,
            backgroundImage.width, backgroundImage.height,
            0, 0, width, height);
    }
}

async function predictWebcam() {
    console.log("Here");
    while (true) {
        // Capture the frame from the webcam.
        const img = await getImage();
        // Only Render on alternate frames
        await predictSegmentation(img)
        img.dispose()
        // raw.dispose()
        frames += 1;
        await tf.nextFrame();
    }
}

/**
 * Captures a frame from the webcam and normalizes it between -1 and 1.
 * Returns a batched image (1-element batch) of shape [1, w, h, c].
 */
async function getImage() {
    // console.log(video, modelInputShape);
    const img = tf.tidy(() => tf.browser.fromPixels(video).resizeBilinear(modelInputShape).expandDims(0).toFloat().div(255));

    // scaleDownCanvasCtx.drawImage(originalVideoElement, 0, 0, width, height, 0, 0, modelWidth, modelHeight);
    // const img = tf.tidy(() => tf.browser.fromPixels(scaleDownCanvas).expandDims(0).toFloat().div(255));

    return img;
}

async function init() {
    // Store the resulting model in the global scope of our app.
    model = await tf.loadGraphModel('model_f16/model.json');
    modelInputShape = model.inputs[0].shape;
    modelInputShape = [modelInputShape[1], modelInputShape[2]]
    demosSection.classList.remove('invisible');
    setInterval(timer, 1000);
}

tf.ready().then(() => init());

// tf.setBackend('webgl').then(() => init());
// tf.setBackend('wasm').then(() => init());