#include <cstdio>
#include <vector>
#include <algorithm>
#include <assert.h>
#include <iostream>
#include <string>
#include <ctime>
#include <random>

using namespace std;

#define enp	puts("****chkchkchk******")

struct proc_info
{
	int type = -1;
	int kernelSize = 0;
	int nxtNum = 0;
	int poolingSize = 0;
	int stride = 0;
	int paddingSize = 0;

	proc_info(int _type, int _kernelSize, int _nxtNum, int _poolingSize, int _stride, int _paddingSize)
	{
		type = _type;
		kernelSize = _kernelSize;
		nxtNum = _nxtNum;
		poolingSize = _poolingSize;
		stride = _stride;
		paddingSize = _paddingSize;
	}
};

enum
{
	Conv,
	Pool
};

typedef vector <proc_info> procedure;

struct convNet_2D
{
	typedef vector <double> mat;
	typedef vector <mat>	mat2D;

	struct MultilayerPerceptron
	{
		vector <double> output;
		vector <int> & numOfUnits;
		vector < vector < vector <double> > > weights;
		vector < vector <double> > units;
		vector < vector <double> > in;
		vector <double> neededOut;
		vector <int> isThereBias;
		double learningRate;
		int numOfLayer;
		vector < vector <double> > delta;

		// _numOfUnits : Layer 별 unit의 수 ex) { 3,4,2 } 이면 입력은 3개, 아웃풋은 2개 히든 레이어 유닛은 4개
		// path : 가중치를 저장해놓은 파일이 있는 주소
		MultilayerPerceptron(vector <int> & _numOfUnits, string path = "") :numOfUnits(_numOfUnits)
		{
			numOfLayer = _numOfUnits.size();

			for (int i = 0; i < numOfLayer - 1; i++)isThereBias.push_back(1);
			isThereBias.push_back(0);

			weights.resize(numOfLayer - 1);
			for (int i = 1; i < numOfLayer; i++)
			{
				weights[i - 1].resize(_numOfUnits[i]);
				for (int j = 0; j < weights[i - 1].size(); j++)
					weights[i - 1][j].resize(_numOfUnits[i - 1] + isThereBias[i - 1]);
			}

			// 가중치를 저장한 파일의 경로가 있으면 불러오고 아니면 랜덤으로 수를 할당
			if (path == "")
				initWeights();
			else
				loadWeights(path);
		}

		void fit(const vector <double>& xTrain, const vector <double> & yTrain, double _learningRate, const vector <double> & init_in)
		{
			learningRate = _learningRate;

			neededOut = yTrain;
			units.clear();
			units.resize(numOfLayer);
			in.clear();
			in.resize(numOfLayer);
			in[0] = init_in;

			for (int j = 0; j < xTrain.size(); j++)
				units[0].push_back(xTrain[j]);// , printf("%lf\n", xTrain[j]);
			
			if(isThereBias[0])
				units[0].push_back(1.0);

			Learning();
		}

		void getOutput(const vector <double> & xTest)
		{
			units.clear();
			units.resize(numOfLayer);
			in.clear();
			in.resize(numOfLayer);

			for (int i = 0; i < xTest.size(); i++)units[0].push_back(xTest[i]);
			if (isThereBias[0])units[0].push_back(1.0);

			forward(units);
			output.clear();

			for (int i = 0; i < units[numOfLayer - 1].size(); i++)output.push_back(units[numOfLayer - 1][i]);
		}

		void saveWeights(const string & path)
		{
			FILE * fp = fopen(path.c_str(), "w");

			for (int i = 0; i < weights.size(); i++)
				for (int j = 0; j < weights[i].size(); j++)
					for (int k = 0; k < weights[i][j].size(); k++)
						fprintf(fp, "%lf ", weights[i][j][k]);

			fclose(fp);
		}
		
		void loadWeights(const string & path)
		{
			FILE * fp = fopen(path.c_str(), "r");

			for (int i = 0; i < weights.size(); i++)
				for (int j = 0; j < weights[i].size(); j++)
					for (int k = 0; k < weights[i][j].size(); k++)
						fscanf(fp, "%lf", &weights[i][j][k]);

			fclose(fp);
		}
		
		void initWeights()
		{
			default_random_engine generator;
			normal_distribution<double> distribution(0.0,0.01);
				
			for (int i = 0; i < weights.size(); i++)
				for (int j = 0; j < weights[i].size(); j++)
					for (int k = 0; k < weights[i][j].size(); k++)
						weights[i][j][k] = distribution(generator);
		}

		double sigmoid(const double & x)
		{
			return 1 / (1 + exp(-x));
		}

		double relu(const double & x)
		{
			return max(x, 0.0);
		}
		
		double dRelu(const double & x)
		{
			if (x > 0)return 1.0;
			else return 0.0;
		}

		double dSigmoid(const double & x)
		{
			double tmp = sigmoid(x);
			return (1 - tmp)*tmp;
		}

		double getInnerProduct(const vector <double> & W, const vector <double> & X)
		{
			double ret = 0;
			for (int i = 0; i < W.size(); i++)ret += W[i] * X[i];
			return ret;
		}

		void calcOneLayer(vector <double> & out, const vector < vector <double> > & W, const vector <double> & input, bool isThereBias, int curLayer)
		{
			in[curLayer].resize(W.size());
			for (int i = 0; i < W.size(); i++)
			{
				in[curLayer][i] = getInnerProduct(W[i], input);
				double tmp = (curLayer == numOfLayer - 1) ? in[curLayer][i] : relu(in[curLayer][i]);
				out.push_back(tmp);
			}

			if (isThereBias)out.push_back(1.0);
		}

		void forward(vector < vector <double> > & _units)
		{
			for (int i = 1; i < numOfLayer; i++)
				calcOneLayer(_units[i], weights[i - 1], _units[i - 1], isThereBias[i], i);

			double sum = 0;
			for (int i = 0; i < _units[numOfLayer - 1].size(); i++)
				sum += exp(in[numOfLayer - 1][i]);

			for (int i = 0; i < _units[numOfLayer - 1].size(); i++)
				_units[numOfLayer - 1][i] = exp(in[numOfLayer - 1][i])/sum;
		}

		void backward()
		{
			delta.clear();
			delta.resize(numOfLayer);

			// 초기 기울기 값을 생성
			for (int i = 0; i < units[numOfLayer - 1].size(); i++)
				delta[numOfLayer - 1].push_back(units[numOfLayer - 1][i]*(1 - units[numOfLayer - 1][i])
			*(neededOut[i] - units[numOfLayer - 1][i]));//, printf("%lf\n", delta[numOfLayer - 1].back());

			
/*			for (int i = 0; i < units[numOfLayer - 1].size(); i++)
				delta[numOfLayer - 1].push_back(dSigmoid(in[numOfLayer - 1][i])*(neededOut[i] - units[numOfLayer - 1][i]));//, printf("%lf %lf %lf\n", log(units[numOfLayer - 1][i]), log(1 - units[numOfLayer - 1][i]), in[numOfLayer - 1][i]);
*/
			for (int l = numOfLayer - 2; l >= 0; l--)
			{
				for (int j = 0; j < units[l].size(); j++)
				{
					if (j < units[l].size() - 1 || (isThereBias[l] == 0 && j == units[l].size() - 1))
					{
						// 기울기값인 delta 를 생성
						double tmp = 0;
						for (int i = 0; i < units[l + 1].size() - isThereBias[l + 1]; i++)
							tmp += weights[l][i][j] * delta[l + 1][i];
						delta[l].push_back(dRelu(in[l][j])*tmp);
					}

					// 가중치 업데이트
					for (int i = 0; i < units[l + 1].size() - isThereBias[l + 1]; i++)
						weights[l][i][j] += learningRate*units[l][j] * delta[l + 1][i];
				}
			}
		}

		void Learning()
		{
			forward(units);
			backward();
		}
	};

	struct ConvolutionLayer
	{
		procedure proc;
		vector < vector < mat2D > > delta;
		vector < vector < vector < mat2D > > > kernel;
		vector < vector < mat2D > > featureMap;
		vector < vector < mat2D > > actFeatureMap;
		vector < vector < mat2D > > bias;
		vector < vector < vector < vector <int > > > > poolY, poolX;
		double learningRate;

		ConvolutionLayer(const procedure & _proc, int _insz ,const string & path = "")
		{
			proc = _proc;
			kernel.resize(proc.size());
			bias.resize(proc.size() + 1);
			int cur = _insz;
			
			for (int i = 0; i < proc.size(); i++)
			{
				if (proc[i].type == Conv)
				{
					cur -= proc[i].kernelSize - 1 - 2*proc[i].paddingSize;

					kernel[i].resize(proc[i].nxtNum, vector <mat2D>(i == 0 ? 1 : proc[i - 1].nxtNum, 
					mat2D(proc[i].kernelSize, mat(proc[i].kernelSize))));
					
					bias[i + 1].resize(proc[i].nxtNum, mat2D(cur, mat(cur)));
				}
				else if (proc[i].type == Pool)
				{
					cur /= proc[i].poolingSize;
				}
			}
			
			if (path != "")
				loadKernel(path);
			else
				initKernel();
		}
		
		void initKernel()
		{
			default_random_engine generator;
			normal_distribution<double> distribution(0.0, 0.01);
			
			for (int i = 0; i < proc.size(); i++)
			{
				for (int j = 0; j < kernel[i].size(); j++)
					for (int k = 0; k < kernel[i][j].size(); k++)
						for (int p = 0; p < kernel[i][j][k].size(); p++)
							for (int q = 0; q < kernel[i][j][k][p].size(); q++)
								kernel[i][j][k][p][q] = distribution(generator);

				for (int j = 0; j < bias[i + 1].size(); j++)
					for (int k = 0; k < bias[i + 1][j].size(); k++)
						for(int p = 0; p < bias[i + 1][j][k].size(); p++)
							bias[i + 1][j][k][p] = distribution(generator);
			}
		}
		
		void loadKernel(const string & path)
		{
			FILE *fp = fopen(path.c_str(), "r");
			for (int i = 0; i < proc.size(); i++)
				for (int j = 0; j < kernel[i].size(); j++)
					for (int k = 0; k < kernel[i][j].size(); k++)
						for (int p = 0; p < kernel[i][j][k].size(); p++)
							for (int q = 0; q < kernel[i][j][k][p].size(); q++)
								fscanf(fp, "%lf", &kernel[i][j][k][p][q]);

			for (int i = 1; i < proc.size() + 1; i++)
				for (int j = 0; j < bias[i].size(); j++)
					for (int k = 0; k < bias[i][j].size(); k++)
						for(int p = 0; p < bias[i][j][k].size(); p++)
							fscanf(fp, "%lf", &bias[i][j][k][p]);
			fclose(fp);
		}

		void saveKernel(const string & path)
		{
			FILE *fp = fopen(path.c_str(), "w");
			for (int i = 0; i < proc.size(); i++)
				for (int j = 0; j < kernel[i].size(); j++)
					for (int k = 0; k < kernel[i][j].size(); k++)
						for (int p = 0; p < kernel[i][j][k].size(); p++)
							for (int q = 0; q < kernel[i][j][k][p].size(); q++)
								fprintf(fp, "%lf ", kernel[i][j][k][p][q]);

			for (int i = 1; i < proc.size() + 1; i++)
				for (int j = 0; j < bias[i].size(); j++)
					for (int k = 0; k < bias[i][j].size(); k++)
						for(int p = 0; p < bias[i][j][k].size(); p++)
							fprintf(fp, "%lf ", bias[i][j][k][p]);
			fclose(fp);
		}

		mat2D conv2D(const mat2D & image, const mat2D & kernel, int stride = 1, int padding_size = 0)
		{
			int st = kernel.size() - 1 - padding_size;
			int en = image.size() + padding_size;
			int sz = en - st;

			mat2D ret(sz, mat(sz, 0.0));

			for (int i = st; i < en; i += stride)
				for (int j = st; j < en; j += stride)
					for (int p = 0; p < kernel.size(); p++)
						for (int q = 0; q < kernel[p].size(); q++)
							if (i - p >= 0 && i - p < image.size() && j - q >= 0 && j - q < image.size())
								ret[i - st][j - st] += image[i - p][j - q] * kernel[p][q];

			return ret;
		}

		double relu(const double & x)
		{
			return max(x, 0.0);
		}

		double dRelu(const double & x)
		{
			if (x > 0)return 1.0;
			else 0.0;
		}

		double sigmoid(const double & x)
		{
			return 1.0/(1.0 + exp(-x));
		}

		double dSigmoid(const double & x)
		{
			double tmp = sigmoid(x);
			return tmp*(1 - tmp);
		}
		
		mat2D flip(const mat2D & kernel)
		{
			mat2D ret = kernel;

			for (int i = 0; i < ret.size(); i++)
				reverse(ret[i].begin(), ret[i].end());
			
			for (int i = 0; i < ret[0].size(); i++)
				for (int j = 0; j < ret.size()/2; j++)
					swap(ret[j][i], ret[ret.size() - 1 - j][i]);

			return ret;
		}

		void sumMat2D(mat2D & m1, const mat2D & m2)
		{
			for (int i = 0; i < m1.size(); i++)
				for (int j = 0; j < m1[i].size(); j++)
					m1[i][j] += m2[i][j];
		}

		void forward(const mat2D & X)
		{
			featureMap.clear();
			featureMap.resize(proc.size() + 1);
			featureMap[0] = vector <mat2D>{ X };
			actFeatureMap.clear();
			actFeatureMap.resize(proc.size() + 1);
			actFeatureMap[0] = vector <mat2D>{ X };
			poolY.clear();
			poolX.clear();
			poolY.resize(proc.size());
			poolX.resize(proc.size());
			
			for (int i = 0; i < proc.size(); i++)
			{
				if (proc[i].type == Conv)
				{
					featureMap[i + 1].resize(kernel[i].size());
					actFeatureMap[i + 1].resize(kernel[i].size());
					for (int k = 0; k < kernel[i].size(); k++)
					{
						for (int j = 0; j < actFeatureMap[i].size(); j++)
						{
							mat2D tmp = conv2D(actFeatureMap[i][j], kernel[i][k][j],1 , proc[i].paddingSize);
							
							if (actFeatureMap[i + 1][k].size() == 0)actFeatureMap[i + 1][k] = tmp;
							else sumMat2D(actFeatureMap[i + 1][k], tmp);
						}
						
						sumMat2D(actFeatureMap[i + 1][k], bias[i + 1][k]);
						
						featureMap[i + 1][k] = actFeatureMap[i + 1][k];
						
						for (int p = 0; p < actFeatureMap[i + 1][k].size(); p++)
							for (int q = 0; q < actFeatureMap[i + 1][k][p].size(); q++)
								actFeatureMap[i + 1][k][p][q] = relu(actFeatureMap[i + 1][k][p][q]);
					}
				}
				else if (proc[i].type == Pool)
				{
					poolX[i].resize(actFeatureMap[i].size());
					poolY[i].resize(actFeatureMap[i].size());
					
					for (int j = 0; j < actFeatureMap[i].size(); j++)
					{
						int sz = actFeatureMap[i][j].size() / proc[i].poolingSize;
						mat2D tmp(sz, mat(sz, -1e15));
						
						poolX[i][j].resize(sz, vector <int> (sz, 0));
						poolY[i][j].resize(sz, vector <int> (sz, 0));
						
						for (int k = 0; k < actFeatureMap[i][j].size(); k++)
						{
							int nk = k / proc[i].poolingSize;
							for (int p = 0; p < actFeatureMap[i][j][k].size(); p++)
							{
								int np = p / proc[i].poolingSize;
								if(tmp[nk][np] < actFeatureMap[i][j][k][p])
								{
									tmp[nk][np] = actFeatureMap[i][j][k][p];
									
									poolY[i][j][nk][np] = k;
									poolX[i][j][nk][np] = p;
								}
							}
						}							
						featureMap[i + 1].push_back(tmp);
						actFeatureMap[i + 1].push_back(tmp);
					}
				}
			}
		}
		
		void calcDelta(const mat & initDelta)
		{
			delta.clear();
			delta.resize(proc.size());

			vector <mat2D> tmp;
			for (int i = 0; i < initDelta.size(); i++)
			{
		//		printf("%0.11lf\n", initDelta[i]);
				tmp.push_back({ {initDelta[i]} });
			}

			delta.push_back(tmp);

			for (int i = proc.size() - 1; i >= 0; i--)
			{
				if (proc[i].type == Conv)
				{
					delta[i].resize(featureMap[i].size());
					for (int j = 0; j < featureMap[i].size(); j++)
					{
						for (int k = 0; k < kernel[i].size(); k++)
						{
							mat2D res = conv2D(flip(kernel[i][k][j]), delta[i + 1][k], 1, delta[i + 1][k].size() - 1);
							
							for (int p = 0; p < res.size(); p++)
								for (int q = 0; q < res[p].size(); q++)
									res[p][q] *= dRelu(featureMap[i][j][p][q]);

							if (delta[i][j].size() == 0)delta[i][j] = res;
							else sumMat2D(delta[i][j], res);
						}
					}
				}
				else if (proc[i].type == Pool)
				{
					for (int j = 0; j < featureMap[i].size(); j++)
					{
						mat2D tmp2(featureMap[i][j].size(), mat(featureMap[i][j][0].size(), 0.0));
						
						for (int k = 0; k < featureMap[i + 1][j].size(); k++)
							for (int p = 0; p < featureMap[i + 1][j][k].size(); p++)
								tmp2[poolY[i][j][k][p]][poolX[i][j][k][p]] = delta[i + 1][j][k][p];

						delta[i].push_back(tmp2);
					}
				}
			}
		}

		void updWeights()
		{
			for (int i = proc.size() - 1; i >= 0; i--)
			{
				if (proc[i].type == Conv)
				{
					for (int k = 0; k < kernel[i].size(); k++)
					{
						for (int j = 0; j < bias[i + 1][k].size(); j++)
							for (int p = 0; p < bias[i + 1][k][j].size(); p++)
								bias[i + 1][k][j][p] += learningRate*delta[i + 1][k][j][p];

						for (int j = 0; j < featureMap[i].size(); j++)
						{
							mat2D tmp = conv2D(flip(actFeatureMap[i][j]), delta[i + 1][k],1, proc[i].paddingSize);

							for (int p = 0; p < tmp.size(); p++)
								for (int q = 0; q < tmp[p].size(); q++)
									kernel[i][k][j][p][q] += learningRate*tmp[p][q];
						}
					}
				}
			}
		}

		void backward(const mat & initDelta)
		{
			calcDelta(initDelta);
			updWeights();
		}
	
	};

	vector < MultilayerPerceptron > mv;
	vector < ConvolutionLayer > cv;

	convNet_2D(const string & mlp_path = "", const string & conv_path = "")
	{
		procedure pd;
		pd.push_back(proc_info(Conv, 13, 8, 0, 1, 0));
		pd.push_back(proc_info(Pool, 0, 8, 4, 4, 0));
		pd.push_back(proc_info(Conv, 4, 120, 0, 1, 0));

		vector <int> nu = {120, 84, 10};

		ConvolutionLayer conv(pd, 28, conv_path);
		MultilayerPerceptron mlp(nu, mlp_path);

		mv.push_back(mlp);
		cv.push_back(conv);
	}

	void trainXY(const mat2D & trainX, const mat & trainY, double _learningRate)
	{
		cv[0].learningRate = _learningRate;
		cv[0].forward(trainX);
		mat mInput;
		mat in0;

		int indx = cv[0].proc.size();
		for (int i = 0; i < cv[0].featureMap[indx].size(); i++)
		{
			in0.push_back(cv[0].featureMap[indx][i][0][0]);
			mInput.push_back(cv[0].actFeatureMap[indx][i][0][0]);
		}
		
		mv[0].fit(mInput, trainY, _learningRate, in0);
		cv[0].backward(mv[0].delta[0]);
	}

	void getOutput(const mat2D & testX)
	{
		cv[0].forward(testX);

		mat mInput;

		int indx = cv[0].proc.size();
		for (int i = 0; i < cv[0].actFeatureMap[indx].size(); i++)
			mInput.push_back(cv[0].actFeatureMap[indx][i][0][0]);

		mv[0].getOutput(mInput);
	}
};

vector < vector < vector <double> > > testX;
vector <int> desiredY;
vector < vector < vector <double> > > trainX;
vector < vector <double> > trainY;
int mxvl = 0;

void readTest()
{
	FILE * fp = fopen("test.txt", "r");
	int sz = 10000;

	testX.resize(sz, vector < vector <double> >(28, vector <double>(28)));;
	desiredY.resize(sz);

	int cnt = -1;
	int tcnt = -1;
	int y = 0;
	int x = -1;

	int s;
	while (~fscanf(fp,"%d", &s))
	{
		x++;
		if (x == 28)
		{
			x = 0;
			y++;
		}

		cnt++;
		if (cnt % 785 == 0)
		{
			if (tcnt == sz - 1)break;
			tcnt++;
			desiredY[tcnt] = s;

			y = 0;
			x = -1;
			continue;
		}

		testX[tcnt][y][x] = (double)s;
	}
	fclose(fp);
}

void readTrain()
{
	FILE * fp = fopen("train.txt", "r");
	int sz = 60000;

	trainX.resize(sz, vector < vector <double> > (28, vector <double> (28)));
	trainY.resize(sz, vector <double>(10, 0.0));

	int cnt = -1;
	int tcnt = -1;
	int y = 0;
	int x = -1;
	int s;

	while (~fscanf(fp,"%d", &s))
	{
		x++;
		if (x == 28)
		{
			x = 0;
			y++;
		}

		cnt++;
		if (cnt % 785 == 0)
		{
			if (tcnt == sz - 1)break;
			tcnt++;
			trainY[tcnt][s] = 1.0;

			y = 0;
			x = -1;

			continue;
		}

		trainX[tcnt][y][x] = (double)s;
	}

	fclose(fp);
}

void test(convNet_2D & convNet)
{
	clock_t st = clock();
	int cn = 0;
	int lim = 10000;

	for (int i = 0; i < lim; i++)
	{
	//	printf("i : %d\n", i);
		convNet.getOutput(testX[i]);

		double mxvl = -1e9;
		int indx = -1;

		for (int j = 0; j < 10; j++)
		{
			if (convNet.mv[0].output[j] > mxvl)
			{
				mxvl = convNet.mv[0].output[j];
				indx = j;
			}
	//		printf("	j : %d prob : %lf\n", j, convNet.mv[0].output[j]);
		}
	//	printf("prob : %lf indx : %d desired out : %d\n", mxvl, indx, desiredY[i]);

		if (indx == desiredY[i])cn++;
	}

	if(mxvl < cn)
	{
		mxvl = cn;
		
		convNet.mv[0].saveWeights("bestMvWeights.txt");
		convNet.cv[0].saveKernel("bestCvWeights.txt");
	}
	
	printf("duration : %d ms\n", clock() - st);
	printf("correct ratio : %d/%d best ratio : %d/%d\n", cn, lim, mxvl, lim);
}

void learning(convNet_2D & convNet, int cnt1)
{
//	puts("before save ");
	convNet.mv[0].saveWeights("prev_mvWeights.txt");
//	puts("before cvWeights ");
	convNet.cv[0].saveKernel("prev_cvWeights.txt");

	clock_t st = clock();
	
	int lim = 60000;

	for (int j = 0; j < 1; j++)
	{
		for (int i = 0; i < lim; i++)
		{
			if(i%1000 == 0)printf("j : %d i : %d\n", j, i);
			convNet.trainXY(trainX[i], trainY[i], 0.001);
		}
	}

	printf("duration : %d ms\n", clock() - st);
	
	
	convNet.mv[0].saveWeights("mvWeights.txt");
	convNet.cv[0].saveKernel("cvWeights.txt");
	
	if(cnt1 != -1)
	{
		char mvs[50], cvs[50];
		
		sprintf(mvs, "mvWeights%d.txt", cnt1);
		sprintf(cvs, "cvWeights%d.txt", cnt1);
		
		string ms = mvs;
		string cs = cvs;

	//	puts("before save ");
		convNet.mv[0].saveWeights(ms);
	//	puts("before cvWeights ");
		convNet.cv[0].saveKernel(cs);
	}
}

void conv()
{
//	convNet_2D convNet;
	convNet_2D convNet("mvWeights.txt", "cvWeights.txt");

	readTest();

/*	readTrain();
	for(int cnt = 0; cnt < 100; cnt++)
	{
		printf("cnt : %d\n", cnt);
		learning(convNet, -1);
		test(convNet);
	}*/
	test(convNet);
	
	puts("end");
}


int main()
{
	conv();

	while(1);
	return 0;
}