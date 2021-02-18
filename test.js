model = await tf.loadGraphModel('model.json');
modelInputShape = model.inputs[0].shape;
modelInputShape = [modelInputShape[1], modelInputShape[2]];
webcam = await tf.data.webcam(video);
[raw, img] = await getImage();
await tf.time(() => model.predict(img))
