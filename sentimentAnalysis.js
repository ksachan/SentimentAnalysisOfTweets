/*
Tri Nguyen, Brian Mooney
CS596 Machine Learning
Dr. Liu 
Sentiment Analysis using Support Vector Machine, Neural Network, and Logistic Regression


TWITTER AUTH

Consumer Key (API Key)	1LsrwvTSs3XAWeJE5fFcDp378
Consumer Secret (API Secret)	Vo7sfjoEOD7DP6haSFR2p4AFTE01qxfHTVSA8k1dgdARNlUQw1

Access Token	4329704114-FtRczjVFxQMo6x2tzA2tfyyud1fOchMEjI7xBPq
Access Token Secret	fyGrcotLQq7V1k94gergfJTXFcaHWcYBaV5nM3B5PSyvn
*/

//importing tools
var Twitter = require('twitter');
var svm = require('node-svm');
var LineByLineReader = require('line-by-line');
var tm = require('text-miner');
var so = require('stringify-object');
var events = require('events');
var fs = require('fs');
var q = require('q');
var nn = require('brain');

//MAIN
var trainData;
var trainVocab;
var trainLabels;
var trainingTuple;
var testVector = [];
var testVectorDTM = [];
var predictionValues = [];
var nnPredictValues = [];
var trainVectorLabel = [];
var predictionOutput;
var svmModel;

//processTrainingData('/Users/yelp_labelled.txt');
//processTrainingData('/Users/yelp_labelled2.txt');
processTrainingData('/Users/imdb_labelled.txt');
//processTrainingData('/Users/test');
//processTrainingData('/Users/test2');

function twitterFeed(){
	//taking input
	console.log("Key word to find sentiment: ");
	process.stdin.setEncoding('utf8');

	process.stdin.on('readable', function(input) {
	  var word = process.stdin.read();
	  testVector = [];
	  if (word !== null) {
	  	streamInput(word);
	  }
	});

	process.stdin.on('end', function() {
	  process.stdout.write('end');
	});

	//Authentication with twitter
	var client = new Twitter({
	  consumer_key: "1LsrwvTSs3XAWeJE5fFcDp378",
	  consumer_secret: "Vo7sfjoEOD7DP6haSFR2p4AFTE01qxfHTVSA8k1dgdARNlUQw1",
	  access_token_key: "4329704114-FtRczjVFxQMo6x2tzA2tfyyud1fOchMEjI7xBPq",
	  access_token_secret: "fyGrcotLQq7V1k94gergfJTXFcaHWcYBaV5nM3B5PSyvn"
	});

	//stream and output
	function streamInput(keyWord){ 
		var numberOfTweets = 0; 

			client.stream('statuses/filter', {track: keyWord}, function(stream) {				
			stream.on('data', function(tweet) {
			  	
			  	if(numberOfTweets < 500){ //number of tweets to take in
				    //console.log(tweet.text);
				    if(typeof(tweet.text) == 'string'){
					    testVector.push(tweet.text);
					    numberOfTweets++;
					}
				}

				else{
					eventEmitter.emit('tweetsReady', testVector);
					stopStream(stream);
				}	
			});

			stream.on('error', function(error) {
			    throw error;
			    process.exit(1);
			});
		});
	}


	function stopStream (s){
		s.destroy();
	}
}

var eventEmitter = new events.EventEmitter();

//Calling dtmTrainingData with sentence and labels
eventEmitter.on('processDataDoneTrain', function(processTrainingData){
	trainLabels = processTrainingData.labels;
	//console.log(trainLabels);
	trainVectorLabel.push(trainLabels);
	dtmTrainingData(processTrainingData.data, processTrainingData.labels);
});

//Calling twitterFeed to get the live tweets 
eventEmitter.on('dtmDoneTrain', function(dtmTrainingData){
	trainData = dtmTrainingData.terms;
	trainVocab = dtmTrainingData.termsVocab;
	//console.log(trainData);
	//console.log(trainVectorLabel)
	//console.log(trainVocab);
	twitterFeed();
});

//Calling vectorize to make a DTM with the training vocab
eventEmitter.on('tweetsReady', function(twitters){
	//console.log(testVector)
	// console.log(twitters)
	vectorize(trainVocab, testVector);
});

//Calling getTrainingSet with DTM terms, and labels
eventEmitter.on('vectorizeData', function(vectoredData){
	//console.log(vectoredData)
	testVectorDTM = vectoredData;
	//printData();
	getTrainingSet(trainData, trainLabels);
});

//Calling svmTrain with trainTuples and sentence from twitter.
eventEmitter.on('trainingSet', function(tuplesTrain){
	trainingTuple = tuplesTrain;
	//console.log(trainingTuple, testVectorDTM);
	svmTrain(trainingTuple, testVectorDTM);
	//neuralNetwork(trainData, trainLabels, testVectorDTM);
});

// //calculate sentiment of NN
// eventEmitter.on('nnResult', function(nnData){
// 	nnCalculateData(nnPredictValues);
// });

//calculate sentiment
eventEmitter.on('Result', function(dataSentiment){
	//console.log(predictionValues);
	//console.log(svmModel);
	calculateSentiment(predictionValues);
});

//parsing the data
function processTrainingData(dataSet){
	var arrData = [];
	var arrLabel = [];

    lr = new LineByLineReader(dataSet);

	lr.on('error', function (err) {
    	console.log(err);
	});

	lr.on('line', function (line) {
		line = line.trim(); //remove all white space

		//adding all data with labels.
		if((line[line.length-1] == 0) || (line[line.length-1] == 1)){
			var noLabel = line.substring(0, line.length-1); //remove label from string.
	     	arrData.push(noLabel); //add the data w/ out label to array.
	     	arrLabel.push(line[line.length-1]); //add the label to array. 
	   	}
	});

	lr.on('end', function () {

		//remove quotes 
		for(var i = 0; i < arrLabel.length; i++){
			if(arrLabel[i] == '1'){
				arrLabel[i] = 1;
			}
			else{
				arrLabel[i] = 0;
			}

			if((i+1) == arrLabel.length){
				var returnData = {
					data: arrData,
					labels: arrLabel
				}

				eventEmitter.emit('processDataDoneTrain', returnData);
			}
		}
	});
}

//creating a document-term matrix for the arrData
function dtmTrainingData(data, label){
	var corpus = new tm.Corpus(data);
	var terms = new tm.Terms(corpus);
	//console.log(terms.nTerms + "Terms");
	//console.log(terms.nDocs + "Sentences");
	
	corpus
		.removeInvalidCharacters() 
		.trim() 	//white space at beg. and end.
		.clean()	//Strips extra whitespace from all documents
		.toLower() 	//lower case
		.removeInterpunctuation() 	//remove all punctuation.
		.removeWords(tm.STOPWORDS.EN); 	//remove all stopwords.

	//fill zeros all undefined terms
	terms.fill_zeros();

	var returnDTM = {
		terms: terms.dtm,
		label: label,
		termsVocab: terms.vocabulary
	}

	eventEmitter.emit('dtmDoneTrain', returnDTM);
}

//create dtm out of incoming text
function vectorize(trainVocab, text){
	//var text = ["Starbucks love and hate it."];
	//console.log(trainVocab)
	//console.log(text);
	var i, j, k, q, u;
	var textWordVector = [];
	var completeTextWordVector = [];
	var wordSplitArray = [];

	//split text into array of words
	for(k = 0; k < text.length; k++){
		wordSplitArray.push(text[k].split(" "));
	}

	for(q = 0; q < wordSplitArray.length; q++){	
		textWordVector = [];	
		//comparing training vocab with textArray
		for(i = 0; i < trainVocab.length; i++){
			textWordVector[i] = 0;
			for(j = 0; j < wordSplitArray[q].length; j++){
				if(trainVocab[i] == wordSplitArray[q][j]){
					textWordVector[i] = 1;
					break;
				}
			}
		}
		
		completeTextWordVector.push(textWordVector);	
		if((q+1) >= wordSplitArray.length){
				eventEmitter.emit('vectorizeData', completeTextWordVector);
		}
	}
}

//compose the data into training set. Making tuples of data and label.
function getTrainingSet(dtm, label){
	var i;
	var tuple;
	var result = [];

	for(i = 0; i < dtm.length; i++){
		tuple = [dtm[i], label[i]];
		result.push(tuple);
	}

	eventEmitter.emit('trainingSet', result);
}

//Neural Network
// function neuralNetwork(trainData, trainLabels, textArray){

// 	var nnResult = [];
// 	var i, j, k;

// 	var net = new nn.NeuralNetwork({
// 		errorThresh: 0.005,  // error threshold to reach 
// 	    iterations: 20000,   // maximum training iterations 
// 	    log: true,           // console.log() progress periodically 
// 	    logPeriod: 10,       // number of iterations between logging 
// 	    learningRate: 0.3    // learning rate 
// 	});

// 	//training parameters
// 	for(i = 0; i < trainData.length; i++){
// 		for(j = 0; j < trainLabels.length; j++){
// 			net.train([{inputs: trainData[i], output: [trainLabels[j]]}]);
// 		}

// 		if((i+1) >= trainData.length){ 
// 			for(k = 0; k < textArray.length; k++){
// 				var outPut = net.run(textArray[k]);
// 				//console.log(outPut)
// 				nnPredictValues.push(outPut);

// 				if((k+1) >= textArray.length){
// 					//console.log(nnPredictValues);
// 					eventEmitter.emit('nnResult', outPut);
// 				}
// 			}
// 		}
// 	}
// }

// function nnCalculateData(nnResult){
// 	var i;
// 	var positive = 0;
// 	var negative = 0;

// 	for(i = 0; i<nnPredictValues.length; i++){
// 		if(nnPredictValues[i] >= .5){
// 			positive++;
// 		}
// 		else{
// 			negative++;
// 		}

// 		if((i+1) >= nnPredictValues.length){
// 			console.log(" ");
// 			console.log("===========================================");
// 			console.log(" ");
// 			console.log("*********Neural Network Data Report********");
// 			console.log(" ");
// 			console.log("Positives: " + positive);
// 			console.log("Negatives: " + negative);
// 			console.log(" ");

// 			if(positive > negative){
// 				console.log("Positive sentiment.");
// 				console.log(" ");
// 				positive = 0;
// 				negative = 0;
// 			}

// 			if(positive < negative){
// 				console.log("Negative sentiment.");
// 				console.log(" ");
// 				positive = 0;
// 				negative = 0;
// 			}

// 			if(positive = negative){
// 				console.log("Neutral sentiment.");
// 				console.log(" ");
// 				positive = 0;
// 				negative = 0;
// 			}

// 			console.log("===========================================");
// 		}
// 	}
// }

function svmTrain(trainData, textArray){
	//console.log(trainData);
	//console.log(textArray);
	var i;

	var clf = new svm.SVM({
		svmType: 'C_SVC',
	 	c: [3], 
	    
	    //kernels parameters 
	    kernelType: 'LINEAR',  
	   
	    //training options 
	    kFold: 2,               
	    normalize: true,        
	    reduce: false,           
	    retainedVariance: 0.99, 
	    eps: 1e-6,              
	    cacheSize: 2000,               
	    shrinking : true,     
	    probability : false   

	});

	if(svmModel == null){
		clf.train(trainData).spread(function (model, report) {
			svmModel = model;
			svmReport = report;
			//console.log(svmModel);

			var newClf = svm.restore(svmModel);
			for(i = 0; i < textArray.length; i++){
				var prediction = newClf.predictSync(textArray[i]);
		 		predictionValues.push(prediction);
		 	}

		 	if((i+1) >= textArray.length){
		 		eventEmitter.emit('Result', prediction);
		 	}
		})
	}
	else{

		var newClf = svm.restore(svmModel);
		for(i = 0; i < textArray.length; i++){
			var prediction = newClf.predictSync(textArray[i]);
	 		predictionValues.push(prediction);
	 	}

	 	if((i+1) >= textArray.length){
	 		eventEmitter.emit('Result', prediction);
	 	}
	}
}

function calculateSentiment(svmPredictionValues){
	var positive = 0; 
	var negative = 0; 
	var i;

	for(i = 0; i < svmPredictionValues.length; i++){	
		if(svmPredictionValues[i] == 1){
			positive++;
		}

		else{
			negative++;
		}

		if((i+1) >= svmPredictionValues.length){
			console.log(" ");
			console.log(" ");
			console.log("===========================================");
			console.log(" ");
			console.log("**************SVM Data Report**************")
			console.log(" ");
			console.log("Positives: " + positive);
			console.log("Negatives: " + negative);
			console.log(" ");
			console.log(svmReport);
			console.log(" ");
			console.log("===========================================");

			if(positive > negative){
				console.log("Positive sentiment.");
				console.log(" ");
				positive = 0;
				negative = 0;
			}

			if(positive < negative){
				console.log("Negative sentiment.");
				console.log(" ");
				positive = 0;
				negative = 0;
			}

			if(positive = negative){
				console.log("Neutral sentiment.");
				console.log(" ");
				positive = 0;
				negative = 0;
			}
		}
	}

	twitterFeed();
	predictionValues.length = 0;
}

// function printData(){
// 	var file = fs.createWriteStream('/Users/data.txt');
// 	file.on('error', function(err) {  //error handling  
// 	});
// 	trainData.forEach(function(v) { file.write(v.join(', ') + ']\n['); });
// 	console.log("copy done")
// 	file.end()
// 	}







