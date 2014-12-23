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

/** 
 * The main class that calls up BPSO for hybrid sampling.
 *
 * @author Pengyi Yang (yangpy@it.usyd.edu.au); Liang Xu (sumkire@swu.edu.cn)
 * @version $Revision: 1.0 $
 */
public class SSOSampling {
	
	/** 
	 * main program of BPSO based hybrid sampling
	 * 
	 * @param args
	 */
	public static void main (String[] args) {
		String fileName = null;
		int iteration = 100;
		int popSize = 60;
		boolean verbose = false;
		
		try {
			//--------- input parameter ----------//
			if (args.length == 0) {
				usage();
			}
			for (int i = 0; i < args.length; i++) {
				if (args[i].equalsIgnoreCase("-h") || 
					args[i].equalsIgnoreCase("h") || 
					args[i].equalsIgnoreCase("-help") || 
					args[i].equalsIgnoreCase("help")) {
					usage();
				}
				if (args[i].equals("-f")) {
					fileName = args[i+1];
					if (fileName.equals(null)) {
						usage();
					}
				}
				if (args[i].equals("-i")) {
					iteration = Integer.parseInt(args[i+1]);
				}
				if (args[i].equals("-p")) {
					popSize = Integer.parseInt(args[i+1]);
				}
				if (args[i].equals("-v")) {
					verbose = true;
				}
			}
			
			// timing the process
			long t0 = System.currentTimeMillis();			
			
			// applying BPSO based sampling
			BPSO bpso = new BPSO(fileName, iteration, popSize, verbose);
			bpso.underSampling();
			
			// stop timing and print the time spent
			long t1 = System.currentTimeMillis();
			long runTime = t1 - t0;
			
			//------------ format time ------------//
			// to seconds
			runTime /= 1000;
			// to minutes
			long sec = runTime % 60;
			long min = runTime / 60;
			System.out.println();
			System.out.println("time spent = " + min + " mins " + sec + " secs");
			System.out.println("SSO sampling accomplished.");
			
		}
		catch (ArrayIndexOutOfBoundsException iobe) {
			usage();
		}
		catch (NullPointerException npe) {
			usage();
		}
		catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	public static void usage () {
		
		System.out.println("\n>>>>>>>>>>>>>>>>>>>>>>>> Pengyi Yang & Liang Xu <<<<<<<<<<<<<<<<<<<<<<<\n"+"*\n"+
							"*                       Welcome to SSO Sampling\n"+
							"*     (A Sample Subset Optimization (SSO)-based Sampling System) \n*\n"+
							"* Copyright (C) GNU General Public License\n"+
							"* Email:	yangpy@it.usyd.edu.au; sumkire@swu.edu.cn\n"+
							"* Institute:\tUniversity of Sydney, Australia; National ICT Australia\n"+
							"* Version:	1.1 (22 Dec. 2011)\n*");

		System.out.println("======================================================================\n"+"*\n"+
							"* General description:\n"+ 
							"*   This is a hybrid algorithm which uses internal cross-validation and \n" +
							"* sample subset optimization technique to rank samples from the majority \n" +
							"* class. The top ranked samples are combined with samples from the minority \n" +
							"* class to create an optimally balanced dataset. "+
				   			"\n======================================================================\n*");		
	
	
		System.out.println("* Usage:\n"+
							"* \tjava -jar SSOSampling.jar -f <dataset.arff> [options]\n*");

		System.out.println("* Dataset:\n"+
				   			"* \t<dataset>\t->\t the selection data is the mining matrix\n"+
				   			"* \t\t\t\t in ARFF format\n"+
				   			"* \t\t\t\t (see ARFF format for more details)");

		System.out.println("* General options:\n"+
				   			"* \t-h\t\t->\t print this help\n"+
				   			"* \t-i <int>\t->\t specify iteration (default=100)\n"+
				   			"* \t-p <int>\t->\t specify population (default=60)\n"+
				   			"* \t-v\t\t->\t run in verbose mode\n*");
		
		System.out.println("* Citations: \n"+
				   			"*  	P. Yang, L. Xu, B. Zhou, Z. Zhang, A. Zomaya \"A Particle Swarm Based \n"+ 
				   			"*   	Hybrid System for Imbalanced Medical Data Sampling\", BMC Genomics,\n"+
				   			"*       10:S34, 2009 \n"+ 
				   			"* \n"+
				   			"*  	P. Yang, Z. Zhang, B. Zhou, A. Zomaya, \"Sample subsets optimization \n" +
				   			"*  	for classifying imbalanced biological data\", In: Proceedings of the \n"+
				   			"*  	15th Pacific-Asia Conference on Knowledge Discovery and Data Mining\n"+
				   			"*  	(PAKDD), LNAI 6635, 333-344, 2011 \n*");	

		System.out.println("* \n>>>>>>>>>>>>>>>>>>>>>>>> Pengyi Yang & Liang Xu <<<<<<<<<<<<<<<<<<<<<<<\n");
		
		System.exit(1);
	}
}
