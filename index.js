const express = require('express');
const app = express();
const path = require('path');
app.use('templates', express.static(path.join(__dirname + '/Users/kainspraveen/Documents/GitHub/cpu-tracking/')));
//app.set('temp', path.join(__dirname, 'templates'));
//app.set('view engine', 'ejs');
app.get('/', function(req, res) {
	res.sendFile(__dirname+"/templates/home.html");
	console.log("time is " + Date.now());
});

app.get('/predicted.html', function(req,res) {
  res.sendFile(__dirname+"/templates/predicted.html");
	console.log("time is " + Date.now());
});
app.get('/realtime.html', function(req,res) {
  res.sendFile(__dirname+"/templates/chart.html");
});


app.listen(3000);
