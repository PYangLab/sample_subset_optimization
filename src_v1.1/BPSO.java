/*
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 2 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program; if not, write to the Free Software
 *    Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

/*
 *    BPSO.java
 *    Copyright (C) 2010 University of Sydney, NSW, Australia
 *
 */

package au.edu.usyd.it.yangpy.sampling;

import java.util.*;
import java.text.DecimalFormat;
import java.io.*;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.Logistic;
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.*;


/** 
 * The class that implements binary particle swarm optimization.
 *
 * History: 
    <Date>     <Author>     <Modification> 
   26/12/2008   Pengyi Yang   Implement the BPSO prototype
   26/02/2009   Liang Xu      Fix some bugs(|v|<Vmax)
   05/03/2009   Pengyi Yang   Modify the algorithm for imbalanced data sampling
   07/03/2009   Liang Xu      Adjust parameters
   30/03/2009   Liang Xu      Refine the BPSO class
   12/04/2009   Pengyi Yang   annotate the source
   17/07/2009   Pengyi Yang	  validate the correctness of the functions
 *
 *
 * @author Pengyi Yang (yangpy@it.usyd.edu.au); Liang Xu (sumkire@swu.edu.cn)
 * @version $Revision: 1.0 $
 */
public class BPSO {
	
	// Data set parameter
	/** entire input set */
	private Instances dataset;
	
	/** internal training set */
	private Instances internalTrain;
	
	/** internal test set */
	private Instances internalTest;
	
	
	// Class parameter
	/** samples from major class */
	private String[] major;
	
	/** class label of major class */
	private int majorLabel;
	
	/** size of the major class */
	private int majorSize;
	
	
	// Particle Swarm Optimization parameter
	/** iteration of BPSO (default = 100)*/
	private int iteration;
	
	/** population size of BPSO (default = 60)*/
	private int popSize;
	
	/** w is the inertia weight */
	private final double w = 0.689343;
	
	/** c1 and c2 are the cognitive and social acceleration constants, respectively */
	private final double c1 = 1.42694;
	private final double c2 = c1;

	/** velocity bound */
	private final double vMax = 0.9820; 
	private final double vMin = 0.0180; 
	
	/** particles */
	private int[][] particles;
	
	/** best position so far of each particle given as fitness */
	private double[] localBest;
	
	/** best position so far of the entire swarm given as fitness */
	private double globalBest;
		
	/** current velocities associate with each particle */
	private double[][] velocity;
	
	/** best position so far of each particle (local optimal) */
	private int[][] localBestParticles;

	/** best position so far of the swarm (global optimal) */
	private int[] globalBestParticle;
	
	/** the search space of the solution */
	private double[][] searchSpace; 
	
	// sampling variables 
	/** the favored samples from major class */
	private ArrayList<String> selectedSample;
	
	/** the tournament selection size */
	//private int tournamentSize;
	
	// Processing variables
	private boolean verbose;
	private Random rand;
	private double avgFitness;
	private DecimalFormat dec; 
	
	
	/**
	 * constructor of BPSO
	 * 
	 * @param fileName	input data set
	 * @param detail	printing mode
	 */
	public BPSO (String fileName, int iteration, int popSize, boolean detail) {
		
		// initialize PSO parameters
		this.iteration = iteration;
		this.popSize = popSize;
		this.verbose = detail;
		rand = new Random(System.currentTimeMillis());
		avgFitness = 0.0;
		selectedSample = new ArrayList<String>();
		//tournamentSize = 2;
		
		// class ratio variables
		double c1 = 0.0;
		double c2 = 0.0;
		double ratio = 0.0;
		
		// load in the imbalanced data set
		try {
			dataset = new Instances(new BufferedReader(new FileReader(fileName)));
			dataset.setClassIndex(dataset.numAttributes() - 1);
		} catch (IOException ioe) {
			ioe.printStackTrace();
		}
		
		// calculate the imbalanced ratio
		for (int i = 0; i < dataset.numInstances(); i++) {
			if (dataset.instance(i).classValue() == 0) {
				c1++;
			}
			else {
				c2++;
			}
		}
		
		if (c1 > c2) {
			majorLabel = 0;
			ratio = c2 / (c1 + c2);
		} 
		else { 
			majorLabel = 1;
			ratio = c1 / (c1 + c2);
		}
			
		System.out.println("-------------------- data stats ----------------------");
		System.out.println("sample of class 0: " + c1);
		System.out.println("sample of class 1: " + c2);
		System.out.println("minority class ratio: " + ratio);
	}
	
	/**
	 * this method starts the under sampling procedure
	 */
	public void underSampling () {
		// create a copy of original data set for cross validation
		Instances randData = new Instances(dataset);
		
		// dividing the data set to 3 folds
		randData.stratify(3);
		
		for (int fold = 0; fold < 3; fold++) {
			// using the first 2 folds as internal training set. the last fold as the internal test set.
			internalTrain = randData.trainCV(3, fold);
			internalTest = randData.testCV(3, fold);

			// calculate the number of the major class samples in the internal training set
			majorSize = 0;
			
			for (int i = 0; i < internalTrain.numInstances(); i++) {
				if (internalTrain.instance(i).classValue() == majorLabel) {
					majorSize++;
				}
			}
		
			// class variable initialization
			dec = new DecimalFormat("##.####");
			localBest = new double[popSize];
			localBestParticles = new int[popSize][majorSize];
			globalBest = Double.MIN_VALUE;
			globalBestParticle = new int[majorSize];
			velocity = new double[popSize][majorSize];
			particles = new int[popSize][majorSize];
			searchSpace = new double[popSize][majorSize];
		
			System.out.println("-------------------- parameters ----------------------");
			System.out.println("CV fold = " + fold);
			System.out.println("inertia weight = " + w);
			System.out.println("c1,c2 = " + c1);
			System.out.println("iteration time = "  + iteration);
			System.out.println("population size = "  + popSize);
			
			// initialize BPSO
			initialization();
			
			// perform optimization process
			findMaxFit();
		
			// save optimization results to array list
			saveResults();
		}
		
		// rank the selected samples and build the balanced dataset
		try {
			createBalanceData();
		} 
		catch (IOException ioe) {
			ioe.printStackTrace();
		}

	}
	
	/**
	 * initiate the particle positions and their velocities
	 */
	public void initialization () {			
		// initiate particle position
		for (int x = 0; x < popSize; x++) {
			for (int y = 0; y < majorSize; y++) {
				
				if (rand.nextDouble() > 0.5) {
					particles[x][y] = 1;
				} else {
					particles[x][y] = 0;
				}
				
				searchSpace[x][y] = 0.0;
			}
		}

		// initiate globalBest and localBest
		for (int count = 0; count < popSize; count++) {
			localBest[count] = Double.MIN_VALUE;
		}
		
		globalBest = Double.MIN_VALUE;
		
		if (verbose == true) {
			System.out.println("------------------ initialization -------------------");
			printMatrix(particles);
		}
			
	}
	
	/**
	 * finding the best results with the given iterations
	 */
	public void findMaxFit () {	
		for (int iterator = 0; iterator < iteration; iterator++) {
			System.out.println("------------------------");
			System.out.println("iteration: " + iterator);
			
			// update the local bests and global best positions and their fitness 
			for (int x = 0; x < popSize; x++) {
				// evaluate the fitness of a given particle
				double fitness = evaluate(x);
				
				// update local values
				if (localBest[x] < fitness) {
					for (int y = 0; y < majorSize; y++) {
						localBestParticles[x][y] = particles[x][y];
					}				
					
					localBest[x] = fitness;
				}
				
				// update global values
				if (globalBest < fitness) {
					for (int y = 0; y < majorSize; y++) {
						globalBestParticle[y] = particles[x][y];
					}									
					
					globalBest = fitness;
				}
			}
			
			// update each particle's velocity and position
			for (int x = 0; x < popSize; x++) {
				for (int y = 0; y < majorSize; y++) {			
					// r1 and r2 are the uniform random numbers generated in the range of [0, 1]
					double r1 = rand.nextDouble();
					double r2 = rand.nextDouble();
					
					// updating equations that govern the BPSO
					velocity[x][y] = w * velocity[x][y] 
					               + c1 * r1 * (localBestParticles[x][y] - particles[x][y]) 
					               + c2 * r2 * (globalBestParticle[y] - particles[x][y]);
					
					if (velocity[x][y] > vMax)
						velocity[x][y] = vMax;
					if (velocity[x][y] < vMin)
						velocity[x][y] = vMin;
					
					particles[x][y] = sigmoid(velocity[x][y]);
					
					/*
					searchSpace[x][y] += velocity[x][y];
					
					if (searchSpace[x][y] > vMax)
					{
						searchSpace[x][y] = vMax;
					}
		
					if (searchSpace[x][y] < vMin)
					{
						searchSpace[x][y] = vMin;
					}
					
					particles[x][y] = simgoid(searchSpace[x][y]);
					*/
				}
			}
			
			printCurrentFitness();
			
			if (verbose == true) {
				printMatrix(particles);
			}
		}
	}

	/**
	 * evaluate a given particle
	 * 
	 * @param PId	particle Id
	 * @return	evaluation score
	 */
	public double evaluate (int PId) {
		if (verbose == true) {
			System.out.println("Evaluate particle: (index) " + PId);
		}
			
		// balance training data using particle information
		try {
			modifyData(PId);
		} catch (IOException ioe) {
			ioe.printStackTrace();
		}
		
		return ensembleClassify();
	}
	
	/**
	 * modify the internal training data with the particle information
	 * 
	 * @param PId	particle Id
	 * @throws IOException
	 */
	public void modifyData(int PId) throws IOException {
		// write the data definition
		BufferedWriter fw = new BufferedWriter(new FileWriter("reduced.arff"));
		fw.write(Instances.ARFF_RELATION + " reducedSet");
		fw.newLine();
		fw.newLine();
		
		for (int i = 0; i < internalTrain.numAttributes() - 1; i++) {
			fw.write(internalTrain.attribute(i).toString());
			fw.newLine();
		}
		
		fw.write(internalTrain.classAttribute().toString());
		fw.newLine();
		fw.newLine();
		fw.write(Instances.ARFF_DATA);
		fw.newLine();
		
		
		// copying all minor samples and loading the major samples into "major[]"
		major = new String[majorSize];
		
		int majorIndex = 0;
		
		for (int i = 0; i < internalTrain.numInstances(); i++) {
			if (internalTrain.instance(i).classValue() != majorLabel) {
				fw.write(internalTrain.instance(i).toString());
				fw.newLine();
			}
			else if (internalTrain.instance(i).classValue() == majorLabel) {
				major[majorIndex] = internalTrain.instance(i).toString();
				majorIndex++;
			}
		}
		
		// adding major samples into the file based on the particle information
		for (int j = 0; j < majorSize; j++) {
			if(particles[PId][j] == 1) {
				fw.write(major[j]);
				fw.newLine();
			}
		}
		
		fw.close();
	}
	
	/**
	 * the target function in fitness form
	 * 
	 * @return	classification accuracy
	 */
	public double ensembleClassify () {
		double fitnessValue = 0.0;
		double classifiersScore = 0.0;
		
		/* load in the modified data set */
		try {
			Instances reducedSet = new Instances(new BufferedReader(new FileReader("reduced.arff")));
			reducedSet.setClassIndex(reducedSet.numAttributes() - 1);
			
			// calculating the evaluation values using each classifier respectively
			if (verbose == true) {
				System.out.println();
				System.out.println(" |----------J4.8-----------|");
				System.out.println(" |            |            |");
			}
			J48 tree = new J48();
			classifiersScore = classify(tree, reducedSet, internalTest);		
			fitnessValue += classifiersScore;
				
			if (verbose == true) {
				System.out.println();
				System.out.println(" |-----3NearestNeighbor----|");
				System.out.println(" |            |            |");
			}
			IBk nn3 = new IBk(3);
			classifiersScore = classify(nn3, reducedSet, internalTest);			
			fitnessValue += classifiersScore;

			if (verbose == true) {
				System.out.println();
				System.out.println(" |--------NaiveBayes-------|");
				System.out.println(" |            |            |");
			}
			NaiveBayes nb = new NaiveBayes();
			classifiersScore = classify(nb, reducedSet, internalTest);		
			fitnessValue += classifiersScore;
			
			if (verbose == true) {
				System.out.println();
				System.out.println(" |-------RandomForest------|");
				System.out.println(" |            |            |");
			}
			RandomForest rf5 = new RandomForest();
			rf5.setNumTrees(5);
			classifiersScore = classify(rf5, reducedSet, internalTest);		
			fitnessValue += classifiersScore;

			if (verbose == true) {
				System.out.println();
				System.out.println(" |---------Logistic--------|");
				System.out.println(" |            |            |");
			}
			Logistic log = new Logistic();
			classifiersScore = classify(log, reducedSet, internalTest);		
			fitnessValue += classifiersScore;
			
		} catch(IOException ioe) {
			ioe.printStackTrace();
		}
		
		fitnessValue /= 5;
		
		if (verbose == true) {
			System.out.println();
			System.out.println("Fitness: " + fitnessValue);
			System.out.println("---------------------------------------------------");
		}
		
		return fitnessValue;
	}
	
	/**
	 * this method evaluate a classifier with 
	 * the sampled data and internal test data
	 * 
	 * @param c	classifier
	 * @param train	sampled set
	 * @param test	internal test set
	 * @return	evaluation results
	 */
	public double classify (Classifier c, Instances train, Instances test) {
		double AUC = 0;
		double FM = 0;
		double GM = 0;
		
		try {
			c.buildClassifier(train);
		
			// evaluate classifier
			Evaluation eval = new Evaluation(train);
			eval.evaluateModel(c, test);	

			AUC = eval.areaUnderROC(1);
			FM = eval.fMeasure(1);
			GM = eval.truePositiveRate(0);
			GM *= eval.truePositiveRate(1);
			GM = Math.sqrt(GM);

		} catch(IOException ioe) {
			ioe.printStackTrace();
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		double mean = (AUC + FM + GM) / 3;
		
		if (verbose == true) {
			System.out.print("AUC: " + dec.format(AUC) + " ");
			System.out.print("FM: " + dec.format(FM) + " ");
			System.out.println("GM: " + dec.format(GM));
			System.out.println("      \\       |       /  ");
			System.out.println("        Mean: " + dec.format(mean));
		}
		
		return mean;
	}
	

	/**
	 * update velocity using simgoid function
	 * 
	 * @param velocity	current velocity value
	 * @return	new velocity value
	 */
	public int sigmoid (double velocity) {
		double randomNumber = rand.nextDouble();
		
		if (randomNumber >= 1/( 1 + Math.exp(-velocity))) {
			return 0;
		} else {
			return 1;
		}
	}
	
	
	/**
	 * @return globalBest value
	 */
	public double getMaxValue () {
		return this.globalBest;
	}
	
	
	/**
	 * this method print a given matrix
	 * 
	 * @param matrix
	 */
	public void printMatrix (int[][] matrix) {
		for (int i = 0; i < matrix.length; i++) {
			System.out.println("Particle_" + i);
			for (int j = 0; j < matrix[i].length; j++) {
				System.out.print(matrix[i][j]+" ");
			}
			System.out.println();
		}
		System.out.println();
	}
	
	
	/**
	 * print the fitness value of current population
	 */
	public void printCurrentFitness () {
		avgFitness = 0.0;
		
		for (int i = 0; i < this.popSize; i++) {
			//System.out.println(localBest[i]);
			avgFitness += localBest[i];
		}
		
		System.out.println("population fitness: " + dec.format(avgFitness / popSize));
	}
	
	
	/**
	 * save the favored samples in local best particles
	 * 
	 */
	public void saveResults () {
		for (int i = 0; i < popSize; i++) {
			for (int j = 0; j < majorSize; j++) {
				if (localBestParticles[i][j] == 1) {	
					selectedSample.add(major[j]);
				}	
			}
		}
	}

	/**
	 * create the balanced training data set
	 * 
	 * @throws IOException
	 */
	public void createBalanceData () throws IOException {
		
		int minorSize = 0;
		BufferedWriter bw = new BufferedWriter(new FileWriter("balanceTrain.arff"));
		
		//------------ output training data definition -------------//
		bw.write(Instances.ARFF_RELATION + " balancedTrainSet");
		bw.newLine();
		
		for (int i = 0; i < dataset.numAttributes(); i++) {
			bw.write(dataset.attribute(i).toString());
			bw.newLine();
		}
		
		bw.write(Instances.ARFF_DATA);
		bw.newLine();
		
		//------------ output samples from minor class ------------//
		for (int j = 0; j < dataset.numInstances(); j++) {	
			if (dataset.instance(j).classValue() != majorLabel) {
				minorSize++;
				bw.write(dataset.instance(j).toString());
				bw.newLine();
			}
		}

		//------------ add selected samples into hash table ------------//
		int max = 0;
		Hashtable<String, Integer> ht = new Hashtable<String, Integer>();

		for (int i = 0; i < selectedSample.size(); i++) {
			if(ht.containsKey(selectedSample.get(i))) {
				Integer C = (Integer)ht.get(selectedSample.get(i));
				int c = C.intValue();
				c++;
				
				if (max < c) {
					max = c;
				}
				
				C = new Integer(c);
				ht.put(selectedSample.get(i), C);
			} else {
				ht.put(selectedSample.get(i), 1);
			}
		}
		
		//------------- Sort the hash table and create balanced training data set -------------//
		System.out.println("-------------------- rankings ---------------------");
		System.out.println("sample                              selection count");
		
		int sampleCount = 0;
		int curCount = max;
		while (curCount != 0) {
			Integer ccurCount = new Integer(curCount);
			
			Iterator<String> itr;
			itr = ht.keySet().iterator();		
			while (itr.hasNext()) {
				String key = (String)itr.next(); // sample
				String value = ht.get(key).toString(); // selection count
				
				// iterate through the selected samples and print the sample of current count
				if (value.equals(ccurCount.toString())) {
					System.out.println(key + "\t" + value);
					
					// direct sampling
					if (sampleCount < minorSize) {
						bw.write(key);
						bw.newLine();
						sampleCount++;
					}
				}
			}
			
			curCount--;
		}
		
		// tournament selection sampling
		/*
		while (sampleCount < minorSize)
		{
			int winner = random.nextInt(selectedSample.size());
			int j;
			
			for (int t = 1; t < tournamentSize; t++)
			{
				j = random.nextInt(selectedSample.size());
				
				while (winner == j)
				{
					j = random.nextInt(selectedSample.size());
				}
				
				// compare the rank
				int rank1 = ht.get(selectedSample.get(winner));
				int rank2 = ht.get(selectedSample.get(j));
				
				if (rank2 > rank1)
				{
					winner = j;
				}
			}

			bw.write(selectedSample.get(winner));
			bw.newLine();
			
			sampleCount++;
		}
		*/
		
		bw.close();
		
		System.out.println("balanced traning dataset created as `balanceTrain.arff'");
		System.out.println("-------------------- -------------- ---------------------");
	}
}
