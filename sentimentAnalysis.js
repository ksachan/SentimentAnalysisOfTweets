/*
Tri Nguyen
Brian Mooney
CS596 Machine Learning
Dr. Liu 
Sentiment Analysis using Support Vector Machine

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

//variable and array declaration
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

//Training the data

//processTrainingData('/Users/yelp_labelled.txt');
//processTrainingData('/Users/yelp_labelled2.txt');
//processTrainingData('/Users/imdb_labelled.txt');
processTrainingData('/Users/test');
//processTrainingData('/Users/test2');


//starting the stream of twitter feed 
function twitterFeed(){

	//taking input
	console.log("Key word to find sentiment: ");
	process.stdin.setEncoding('utf8');
	process.stdin.on('readable', function(input) {
		var word = process.stdin.read();
	 
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
			  	
			  	//number of tweets to take in
			  	if(numberOfTweets < 1000){ 
				    console.log(tweet.text);
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
	trainVectorLabel.push(trainLabels);
	dtmTrainingData(processTrainingData.data, processTrainingData.labels);
});

//Calling twitterFeed to get the tweets 
eventEmitter.on('dtmDoneTrain', function(dtmTrainingData){
	trainData = dtmTrainingData.terms;
	trainVocab = dtmTrainingData.termsVocab;
	twitterFeed();
});

//Calling vectorize to make a DTM with the training vocab
eventEmitter.on('tweetsReady', function(twitters){
	vectorize(trainVocab, testVector);
});

//Calling getTrainingSet with DTM terms, and labels
eventEmitter.on('vectorizeData', function(vectoredData){
	testVectorDTM = vectoredData;
	getTrainingSet(trainData, trainLabels);
});

//Calling svmTrain with trainTuples and sentence from twitter.
eventEmitter.on('trainingSet', function(tuplesTrain){
	trainingTuple = tuplesTrain;
	svmTrain(trainingTuple, testVectorDTM);
});

//calculate sentiment
eventEmitter.on('Result', function(dataSentiment){
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

//create document term matrix out of incoming text
function vectorize(trainVocab, text){
	var i, j, k, q, u;
	var textWordVector = [];
	var completeTextWordVector = [];
	var wordSplitArray = [];

	//split text into array of words
	for(k = 0; k < text.length; k++){
		wordSplitArray.push(text[k].split(" "));
	}

	for(q = 0; q < wordSplitArray.length; q++){		
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

//SVM method
function svmTrain(trainData, textArray){
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

//function to tally up the positive an negative from the SVM prediction
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


