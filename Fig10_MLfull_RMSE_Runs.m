% Solve an Input-Output Fitting problem with a Neural Network
% This code was coded by Xiaobing Zhou fusing MATLAB fitnet and train functions.
% 
% Trainingdata.xlsx includes 12 columns of data as input for traning,
% validation, and test. 
% 
disp ('   ')
disp ('This computer is working, please wait ....')

tic; % start recording time
% Clear up
clc; clf; clear; close all;

% delete old output '*_RMSE-R2p.xlsx' files
oldfiles = dir ('*_RMSE-R2-p.xlsx');
[r,c]=size(oldfiles);
if (r ~=0)
    delete *_RMSE-R2-p.xlsx;
end

% batch process input data files: five input files:
% trainingdata_BlackEagle.xlsx, trainingdata_Cochrane.xlsx,
% trainingdata_Morony.xlsx, trainingdata_Rainbow.xlsx,
% trainingdata_Ryan.xlsx.
files = dir ('training*.xlsx');
for k = 1:length(files)
    filename = files(k).name; %filename=trainingdata_BlackEagle.xlsx
    name = split(filename,'.');  %   {'trainingdata_BlackEagle'}    {'xlsx'                   }
    basename = split(name{1},'_'); % basename ={'training' 'BlackEagle'}
    outfilename1 = [basename{2},'_R2-run.xlsx']; % filename for output
    outfilename2 = [basename{2},'_RMSE-R2-p.xlsx']; % filename for output
    str = ['Working on ' basename{2} ' hydropower plant ...'];
    disp(str);

    data = xlsread(filename);

    NN = size(data,2); % number of columns in input data
    x = zeros(size(data(:,1),1),NN);
    x=data;
    x =x'; % input for training, tranposed data
    t = x(NN,:); % last column is the target data

    X = [ones(size(x(1,:)))];
    N=NN-1;

    % rmse_train = zeros(1,N);
    rmse_Train = zeros(1,1);
    rmse_Val = zeros(1,1);
    rmse_Test = zeros(1,1);

    p = zeros(1,N);
    X=x(1:N,:);
    nnet.guis.closeAllViews(); % close all the neural network diagrams created before.
    close all;

    trainFcn = 'trainlm';  % Levenberg-Marquardt backpropagation.
    hiddenLayerSize = N+1;  % setting hidenLayerSize == # of parameters

    Nrun =1000;  % run 1000 trains

    R2Tmax=0.0;
    kmax=0;

    for k=1:Nrun
        net = fitnet(hiddenLayerSize,trainFcn); % create network object and store into "net"
        % Choose Input and Output Pre/Post-Processing Functions
        % For a list of all processing functions type: help nnprocess
        net.input.processFcns = {'removeconstantrows','mapminmax'};
        net.output.processFcns = {'removeconstantrows','mapminmax'};
        net.divideParam.trainRatio = 70/100;  % dividion into 70:15:15
        net.divideParam.valRatio = 15/100;
        net.divideParam.testRatio = 15/100;

        [net, traininfo] = train(net, X, t);  % train function will take first the network to train as

        ytrained = net(X(:,traininfo.trainInd)); % ML model prediction
        yTraintrue = t(traininfo.trainInd); % retrieving target data

        % Preparing output data from training
        mdl=fitlm(yTraintrue,ytrained); % linear regression to see how good the prediction from ML is
        R2T(k)=mdl.Rsquared.Ordinary; % get R2
        pT=mdl.Coefficients.pValue(2); % get p-value
        rmse_Train(k) = mdl.RMSE; % get RMSE

        if (R2T(k) > R2Tmax)  % optimizing ML model by choosing the max R2
            R2Tmax = R2T(k);
            netmax=net;
            traininfomax = traininfo;
            kmax=k;  % which run is the best
            % mdlmax=mdl;
        end

        % Preparing output data from validation
        yVal = net(X(:,traininfo.valInd));
        yValtrue = t(traininfo.valInd);
        mdl=fitlm(yValtrue,yVal);
        R2V(k)=mdl.Rsquared.Ordinary;
        pV=mdl.Coefficients.pValue(2);
        rmse_Val(k) = mdl.RMSE;
        % rmse_val = sqrt(mean((yVal - yValtrue).^2));

        % Preparing output data from test
        yTest = net(X(:,traininfo.testInd));
        yTesttrue = t(traininfo.testInd);
        mdl=fitlm(yTesttrue,yTest);
        R2Test(k)=mdl.Rsquared.Ordinary;
        ptest=mdl.Coefficients.pValue(2);
        rmse_Test(k) = mdl.RMSE;
    end

    % Write output data file R2 & RMSE verus number of runs
    Nr=1:Nrun;
    title = {'Number of runs' 'RMSE_train' 'R2_train' 'RMSE_Val' 'R2_Val' 'RMSE_Test' 'R2_Test' 'Best_Run'};
    xlswrite (outfilename1,title, 'Sheet1', 'A1');
    xlswrite (outfilename1,Nr','Sheet1', 'A2');
    xlswrite (outfilename1,rmse_Train','Sheet1', 'B2');
    xlswrite (outfilename1,R2T','Sheet1', 'C2');
    xlswrite (outfilename1,rmse_Val','Sheet1', 'D2');
    xlswrite (outfilename1,R2V','Sheet1', 'E2');
    xlswrite (outfilename1,rmse_Test','Sheet1', 'F2');
    xlswrite (outfilename1,R2Test','Sheet1', 'G2');
    xlswrite (outfilename1,kmax,'Sheet1', 'H2');

    % Output the best
    netmax.divideParam.trainRatio = 70/100;
    netmax.divideParam.valRatio = 15/100;
    netmax.divideParam.testRatio = 15/100;

    ytrained = netmax(X(:,traininfomax.trainInd)); %
    yTraintrue = t(traininfomax.trainInd);
    mdlmax=fitlm(yTraintrue,ytrained);
    R2T=mdlmax.Rsquared.Ordinary;
    pT=mdlmax.Coefficients.pValue(2);
    rmse_Train = mdlmax.RMSE;

    yVal = netmax(X(:,traininfomax.valInd));
    yValtrue = t(traininfomax.valInd);
    mdlmax=fitlm(yValtrue,yVal);
    R2V=mdlmax.Rsquared.Ordinary;
    pV=mdlmax.Coefficients.pValue(2);
    rmse_Val = mdlmax.RMSE;

    yTest = netmax(X(:,traininfo.testInd));
    yTesttrue = t(traininfo.testInd);
    mdlmax=fitlm(yTesttrue,yTest);
    R2Test=mdlmax.Rsquared.Ordinary;
    ptest=mdlmax.Coefficients.pValue(2);
    rmse_Test = mdlmax.RMSE;

    title = {'RMSE_train' 'R2_train' 'PT-measured' 'PT-predicted' 'RMSE_val' 'R2_val' 'PV-measured' 'PV-predicted' 'RMSE_test' 'R2_test' 'Ptest-measured' 'Ptest-predicted' 'RMSE_all' 'R2_all' 'Pall-measured' 'Pall-predicted'};
    xlswrite (outfilename2,title, 'Sheet1', 'A1');

    xlswrite (outfilename2,rmse_Train','Sheet1', 'A2');
    xlswrite (outfilename2,R2T','Sheet1', 'B2');
    xlswrite (outfilename2,yTraintrue','Sheet1', 'C2');
    xlswrite (outfilename2,ytrained','Sheet1', 'D2');

    xlswrite (outfilename2,rmse_Val','Sheet1', 'E2');
    xlswrite (outfilename2,R2V','Sheet1', 'F2');
    xlswrite (outfilename2,yValtrue','Sheet1', 'G2');
    xlswrite (outfilename2,yVal','Sheet1', 'H2');

    xlswrite (outfilename2,rmse_Test','Sheet1', 'I2');
    xlswrite (outfilename2,R2Test','Sheet1', 'J2');
    xlswrite (outfilename2,yTesttrue','Sheet1', 'K2');
    xlswrite (outfilename2,yTest','Sheet1', 'L2');

    % For all data points
    for k=1:Nrun
        net = fitnet(hiddenLayerSize,trainFcn); % create network object and store into "net"
        % Choose Input and Output Pre/Post-Processing Functions
        % For a list of all processing functions type: help nnprocess
        net.input.processFcns = {'removeconstantrows','mapminmax'};
        net.output.processFcns = {'removeconstantrows','mapminmax'};
        net.divideParam.trainRatio = 100/100;
        net.divideParam.valRatio = 0/100;
        net.divideParam.testRatio = 0/100;

        [net, traininfo] = train(net, X, t);  % train function will take first the network to train as
        ytrained = net(X(:,traininfo.trainInd)); %
        yTraintrue = t(traininfo.trainInd);

        mdl=fitlm(yTraintrue,ytrained);
        R2T(k)=mdl.Rsquared.Ordinary;
        pT=mdl.Coefficients.pValue(2);
        rmse_Train(k) = mdl.RMSE;

        if (R2T(k) > R2Tmax)
            R2Tmax = R2T(k);
            netmax=net;
            traininfomax = traininfo;
            mdlmax=mdl;
        end
    end

    netmax.divideParam.trainRatio = 100/100;
    netmax.divideParam.valRatio = 0/100;
    netmax.divideParam.testRatio = 0/100;

    ytrainedall = netmax(X(:,traininfomax.trainInd)); %
    yTraintrueall = t(traininfomax.trainInd);
    mdlmax=fitlm(yTraintrueall,yTraintrueall);

    R2Tall=mdlmax.Rsquared.Ordinary;
    pTall=mdlmax.Coefficients.pValue(2);
    rmse_Trainall = mdlmax.RMSE;

    xlswrite (outfilename2,rmse_Trainall','Sheet1', 'M2');
    xlswrite (outfilename2,R2Tall','Sheet1', 'N2');
    xlswrite (outfilename2,yTraintrueall','Sheet1', 'O2');
    xlswrite (outfilename2,ytrainedall','Sheet1', 'P2');
end
toc; % End recording time
