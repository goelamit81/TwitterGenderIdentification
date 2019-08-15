package com.upgrad.ml.sparkmlassignment;

import org.apache.spark.ml.feature.StringIndexerModel;

import static org.apache.spark.sql.functions.callUDF;
import static org.apache.spark.sql.functions.col;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.DecisionTreeClassificationModel;
import org.apache.spark.ml.classification.DecisionTreeClassifier;
import org.apache.spark.ml.classification.RandomForestClassificationModel;
import org.apache.spark.ml.classification.RandomForestClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.HashingTF;
import org.apache.spark.ml.feature.IDF;
import org.apache.spark.ml.feature.IndexToString;
import org.apache.spark.ml.feature.Normalizer;
import org.apache.spark.ml.feature.StandardScaler;
import org.apache.spark.ml.feature.StopWordsRemover;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.Tokenizer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.api.java.UDF1;
import org.apache.spark.sql.types.DataTypes;

import static org.apache.spark.sql.functions.regexp_replace;

import java.text.SimpleDateFormat;
import java.util.Date;

import static org.apache.spark.sql.functions.lower;

public class SparkMLClassificationAssignmentDataPreProcessing {

	/*
	 * Setting hyper parameters for Decision Tree:
	 * 
	 * Setting maxDepth as 12 for optimal accuracy. Have checked for all numbers
	 * between - 10 and 20. maxDepth = 12 yields good result for Decision Tree.
	 * Numbers outside range of 10-20, yield poor results.
	 * 
	 * Setting minInfoGain as 0.0 for optimal accuracy. Have checked for 0.0, 0.2
	 * and 0.4. minInfoGain = 0.0 yields good result for Decision Tree
	 * 
	 * Setting maxBins as 9 for optimal accuracy. Have checked for all numbers
	 * between - 2 and 10. maxBins = 9 yields good result for Decision Tree
	 * 
	 * Setting minInstancesPerNode as 8 for optimal accuracy. Have checked for all
	 * numbers between - 2 and 10. minInstancesPerNode = 9 yields good result for
	 * Decision Tree
	 */
	public static int maxDepthDT = 12;
	public static double minInfoGainDT = 0.0;
	public static int maxBinsDT = 9;
	public static int minInstancesPerNodeDT = 9;

	/*
	 * Setting hyper parameters for Random Forest:
	 * 
	 * Setting maxDepth as 18 for optimal accuracy. Have checked for all numbers
	 * between - 10 and 20. maxDepth = 18 yields good result for Random Forest.
	 * Numbers outside range of 10-20, yield poor results.
	 * 
	 * Setting minInfoGain as 0.0 for optimal accuracy. Have checked for 0.0, 0.2
	 * and 0.4. minInfoGain = 0.0 yields good result for Random Forest
	 *
	 * Setting maxBins as 6 for optimal accuracy. Have checked for all numbers
	 * between - 2 and 10. maxBins = 6 yields good result for Random Forest
	 * 
	 * Setting minInstancesPerNode as 8 for optimal accuracy. Have checked for all
	 * numbers between - 2 and 10. minInstancesPerNode = 8 yields good result for
	 * Random Forest
	 */
	public static int maxDepthRF = 18;
	public static double minInfoGainRF = 0.0;
	public static int maxBinsRF = 6;
	public static int minInstancesPerNodeRF = 8;

	// StringIndexerModel for converting gender to numeric
	private static StringIndexerModel indexerModelGender = null;

	// UDF to convert hex to integer and return as string
	private static UDF1<String, String> hexToInteger = new UDF1<String, String>() {

		private static final long serialVersionUID = 1L;

		public String call(String str) throws Exception {
			try {
				return String.valueOf(Integer.parseInt(str, 16));
			} catch (NumberFormatException nfe) {
				return String.valueOf(0);
			}
		}
	};

	public static void main(String[] args) {

		// Setup logging to error only
		Logger.getLogger("org").setLevel(Level.ERROR);
		Logger.getLogger("akka").setLevel(Level.ERROR);

		// Setting up Spark Session
		SparkSession sparkSession = SparkSession.builder().appName("SparkMLClassification").master("local[*]")
				.getOrCreate();

		// Print Start time
		System.out.println("\nStart Time : " + new SimpleDateFormat("yyyy-MM-dd HH:mm:ss").format(new Date()) + "\n");

		// Reading Data from input CSV file Inferring Schema and Setting Header as True
		Dataset<Row> csvData1 = sparkSession.read().option("header", true).option("inferSchema", true).csv(args[0]);

		// Show summary of loaded data
		System.out.println("\nSummary of data read from file as is:\n");
		csvData1.describe().show();

		// Print count of records read from input file
		System.out.println("Total records read from the file : " + csvData1.count() + "\n");

		/*
		 * Reading Data from input CSV file Inferring Schema and Setting Header as True.
		 * Dropping corrupted records.
		 */
		Dataset<Row> csvData = sparkSession.read().option("header", true).option("inferSchema", true)
				.option("parserLib", "univocity").option("mode", "DROPMALFORMED").csv(args[0]);

		// Show summary of loaded data
		System.out.println(
				"\nSummary of data read from file after using mode as DROPMALFORMED and parseLib as univocity is:\n");
		csvData.describe().show();

		// Print count of records read from input file
		System.out.println("Total records read from the file : " + csvData.count() + "\n");

		/*
		 * Fetch specific columns which are found to be useful based on running again
		 * and again with different combination of feature columns.
		 */
		Dataset<Row> twitterData = csvData.select(col("gender"), col("description"), col("link_color"),
				col("sidebar_color"), col("text"), col("tweet_count").cast(DataTypes.DoubleType), col("gender_gold"),
				col("gender:confidence").as("gender_confidence").cast(DataTypes.DoubleType));

		/*
		 * Interested in records where description is present (not null) as need to
		 * calculate TF-IDF. Values in text and description should not be null otherwise
		 * error comes while calculating TF-IDF.
		 * 
		 * Interested in those records where gender is either male , female , brand and
		 * (gender_gold is either male, female , brand or gender_confidence is 1.0).
		 * 
		 * This criteria helps in improving accuracy of the model.
		 * 
		 * Dropping gender_confidence and gender_gold as not needed further.
		 */
		twitterData = twitterData.where(
				"((gender in ('male','female','brand') and gender_confidence = 1.0) or gender_gold in ('male','female','brand'))"
						+ " and description is not null")
				.drop("gender_confidence", "gender_gold");

		// Show some data
		System.out.println("\nShowing some data after filtering:\n");
		twitterData.show(10);

		// Print count of records left
		System.out.println("Total records left after filtering : " + twitterData.count() + "\n");

		/*
		 * Setting up UDF to convert hexadecimal into integers. This UDF will convert
		 * hexadecimal to integers and will return as string. This UDF is used to
		 * convert link_color and sidebar_color hex values.
		 */
		sparkSession.udf().register("toInteger", hexToInteger, DataTypes.StringType);
		twitterData = twitterData.withColumn("link_color_indexed", callUDF("toInteger", twitterData.col("link_color")))
				.drop("link_color");
		twitterData = twitterData
				.withColumn("sidebar_color_indexed", callUDF("toInteger", twitterData.col("sidebar_color")))
				.drop("sidebar_color");

		// Show some data
		System.out.println("\nShowing some data after converting link_color and sidebar_color to integers:\n");
		twitterData.show(10);

		/*
		 * Replace non word characters from text and description columns with space and
		 * convert into lower case.
		 */
		twitterData = twitterData.withColumn("text", lower(regexp_replace(twitterData.col("text"), "[\\W]", " ")));

		twitterData = twitterData.withColumn("description",
				lower(regexp_replace(twitterData.col("description"), "[\\W]", " ")));

		// Show some data
		System.out.println(
				"\nShowing some data after removing non word characters from text and description and converting both to lower case:\n");
		twitterData.show(10);

		/*
		 * Cast link_color_indexed and sidebar_color_indexed to integers
		 */
		twitterData = twitterData.select(col("gender"), col("description"),
				col("link_color_indexed").cast(DataTypes.IntegerType),
				col("sidebar_color_indexed").cast(DataTypes.IntegerType), col("text"), col("tweet_count"));

		// Show some data
		System.out.println("\nShowing some data before building a pipeline:\n");
		twitterData.show(10);

		/*
		 * Lets setup different stages and will use pipeline. All transformations will
		 * be done directly in pipeline.
		 */

		/*
		 * Setup StringIndexerModel to convert String column 'gender' to numeric and
		 * relabel target variable.
		 */
		indexerModelGender = new StringIndexer().setInputCol("gender").setOutputCol("gender_indexed").fit(twitterData);

		// Tokenize text column and set as text_words
		Tokenizer tokenizer_text = new Tokenizer().setInputCol("text").setOutputCol("text_words");

		// Tokenize description column and set as description_words
		Tokenizer tokenizer_desc = new Tokenizer().setInputCol("description").setOutputCol("description_words");

		// Remove stop words from text_words and set as text_removed
		StopWordsRemover remover_text = new StopWordsRemover().setInputCol("text_words").setOutputCol("text_removed");

		// Remove stop words from description_words and set as description_removed
		StopWordsRemover remover_desc = new StopWordsRemover().setInputCol("description_words")
				.setOutputCol("description_removed");

		// Calculate term frequency from text_removed and set as hashingtf_text
		HashingTF tf_text = new HashingTF().setNumFeatures(1000).setInputCol("text_removed")
				.setOutputCol("hashingtf_text");

		// Calculate term frequency from description_removed and set as hashingtf_desc
		HashingTF tf_desc = new HashingTF().setNumFeatures(1000).setInputCol("description_removed")
				.setOutputCol("hashingtf_desc");

		// Calculate inverse document frequency from hashingtf_text and set as idf_text
		IDF idf_text = new IDF().setInputCol("hashingtf_text").setOutputCol("idf_text");

		// Calculate inverse document frequency from hashingtf_desc and set as idf_desc
		IDF idf_desc = new IDF().setInputCol("hashingtf_desc").setOutputCol("idf_desc");

		/*
		 * Setup Vector assembler to assemble all the required columns that will be used
		 * as features. The columns chosen as features will be converted to desired
		 * numeric form in pipeline.
		 */
		VectorAssembler assembler = new VectorAssembler().setInputCols(
				new String[] { "link_color_indexed", "tweet_count", "sidebar_color_indexed", "idf_text", "idf_desc" })
				.setOutputCol("features");

		// Setup StandardScaler in order to scale features and set as scaledFeatures
		StandardScaler scaler = new StandardScaler().setInputCol("features").setOutputCol("scaledFeatures");

		/*
		 * Setup Normalizer in order to normalize scaled features and set as
		 * normalizedFeatures
		 */
		Normalizer normalizer = new Normalizer().setInputCol("scaledFeatures").setOutputCol("normalizedFeatures")
				.setP(2.0);

		/*
		 * Create and Run Pipeline for all stages set so far. Stage set so far, are
		 * common to both Decision Tree and Random Forest models so creating and running
		 * pipeline to get desired data now.
		 */
		Pipeline pipeline = new Pipeline()
				.setStages(new PipelineStage[] { indexerModelGender, tokenizer_text, tokenizer_desc, remover_text,
						remover_desc, tf_text, tf_desc, idf_text, idf_desc, assembler, scaler, normalizer });

		// Fit the pipeline to training data.
		PipelineModel model = pipeline.fit(twitterData);

		// Transform data to obtain final transformed data.
		Dataset<Row> twitterDataTransformed = model.transform(twitterData);

		/*
		 * Split the data randomly in two parts (training and testing) using seed so
		 * split is deterministic.
		 */
		Dataset<Row>[] dataSplit = twitterDataTransformed.randomSplit(new double[] { 0.7, 0.3 }, 46L);
		// Fetch the training data
		Dataset<Row> trainingData = dataSplit[0];
		// Fetch the testing data
		Dataset<Row> testingData = dataSplit[1];

		System.out.println("\nTotal records in trainingData: " + trainingData.count());
		System.out.println("Total records in testingData: " + testingData.count());

		/*
		 * Set up Decision Tree Model with hyper parameters selected from analysis of
		 * various values.
		 */
		DecisionTreeClassifier dt = new DecisionTreeClassifier().setLabelCol("gender_indexed")
				.setFeaturesCol("normalizedFeatures").setMaxDepth(maxDepthDT).setMinInfoGain(minInfoGainDT)
				.setMinInstancesPerNode(minInstancesPerNodeDT).setMaxBins(maxBinsDT).setSeed(46L);

		DecisionTreeClassificationModel modelDT = dt.fit(trainingData);

		Dataset<Row> predictionsDT = null;

		// Predict on training data
		predictionsDT = modelDT.transform(trainingData);

		System.out.println("\nDecision Tree classification model evaluation using training data :\n");
		evaluateModel(predictionsDT);

		// Predict on testing data
		predictionsDT = modelDT.transform(testingData);

		System.out.println("\nDecision Tree classification model evaluation using testing data :\n");
		evaluateModel(predictionsDT);

		/*
		 * Set up the Random Forest Model with hyper parameters selected from analysis
		 * of various values.
		 */
		RandomForestClassifier rf = new RandomForestClassifier().setLabelCol("gender_indexed")
				.setFeaturesCol("normalizedFeatures").setMaxDepth(maxDepthRF).setMinInfoGain(minInfoGainRF)
				.setMinInstancesPerNode(minInstancesPerNodeRF).setMaxBins(maxBinsRF).setSeed(46L);

		RandomForestClassificationModel modelRF = rf.fit(trainingData);

		Dataset<Row> predictionsRF = null;

		// Predict on training data
		predictionsRF = modelRF.transform(trainingData);

		System.out.println("\nRandom forest classification model evaluation using training data :\n");
		evaluateModel(predictionsRF);

		// Predict on testing data
		predictionsRF = modelRF.transform(testingData);

		System.out.println("\nRandom forest classification model evaluation using test data :\n");
		evaluateModel(predictionsRF);

		// Print End time
		System.out.println("\nEnd Time : " + new SimpleDateFormat("yyyy-MM-dd HH:mm:ss").format(new Date()) + "\n");

	}

	private static void evaluateModel(Dataset<Row> predictionData) {

		// Select (prediction, gender_indexed label)
		MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
				.setLabelCol("gender_indexed").setPredictionCol("prediction");

		/*
		 * Transform back numeric gender prediction to string format and create
		 * confusion matrix.
		 */
		IndexToString converter = new IndexToString().setInputCol("prediction").setOutputCol("predicted_gender")
				.setLabels(indexerModelGender.labels());

		Dataset<Row> outputData = converter.transform(predictionData);

		// Compute accuracy
		evaluator.setMetricName("accuracy");
		double accuracy = evaluator.evaluate(outputData);
		System.out.println("Accuracy = " + Math.round(accuracy * 100) + " %");

		// Compute weightedPrecision
		evaluator.setMetricName("weightedPrecision");
		double precision = evaluator.evaluate(outputData);
		System.out.println("Precision = " + Math.round(precision * 100) + " %");

		// Compute weightedRecall
		evaluator.setMetricName("weightedRecall");
		double recall = evaluator.evaluate(outputData);
		System.out.println("Recall = " + Math.round(recall * 100) + " %");

		// Compute F1 score
		evaluator.setMetricName("f1");
		double f1Score = evaluator.evaluate(outputData);
		System.out.println("f1 score = " + Math.round(f1Score * 100) + " %");

		Dataset<Row> confusionMatrix = outputData.groupBy("gender", "predicted_gender").count().orderBy("gender",
				"predicted_gender");

		/*
		 * Display Confusion Matrix
		 */
		System.out.println("\nConfusion Matrix :\n");
		confusionMatrix.show();

	}

}
