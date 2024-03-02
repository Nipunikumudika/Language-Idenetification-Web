const express = require('express');
const { spawn } = require('child_process');
const bodyParser = require('body-parser');
const cors = require('cors');

const app = express();
const port = process.env.PORT || 5000;

app.listen(port, () => {
  console.log(`App listening on port ${port}!`);
});

app.use(cors());
app.use(bodyParser.text()); 

app.get("/", (req, res) => res.send("Express on Vercel"));

app.post('/predict', (req, res) => {
  // Extract raw text from the request body
  const features = req.body;
  console.log('Received features:', features);

  // Spawn a new child process to call the python script
  const python = spawn('python', ['./ml.py', features]);

  let dataToSend = '';

  // Collect data from the script
  python.stdout.on('data', (data) => {
    dataToSend += data.toString();
  });

  // In the close event, we are sure that the stream from the child process is closed
  python.on('close', (code) => {
    console.log(`Child process close all stdio with code ${code}`);
    // Send data to the browser
    res.send(dataToSend);
  });
});


