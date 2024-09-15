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
    outfilename = [basename{2},'_RMSE-R2-p.xlsx']; % filename for output
    str = ['Working on ' basename{2} ' hydropower plant ...'];
    disp(str);
    
    data = xlsread(filename);
    
    NN = size(data,2); % number of columns in input data
    x = zeros(size(data(:,1),1),NN);
    [nr,NN]=size(data);
    x=data';
    t = x(NN,:); % last column is the target data
    % X = [];
    N=NN-1;  %N=12
    rmse_train = zeros(1,N);
    p = zeros(1,N);
    R2Tmax=0.0;
    R2Tall=zeros(1,N);
    %     N=1;
    % Starting sensivity to number of parameters
    for i = 1:N
        % X = [X; x(i,:)];
        X=x(1:i,:);
        % nnet.guis.closeAllViews(); % close all the neural network diagrams created before.
        % close all;
        trainFcn = 'trainlm';  % Levenberg-Marquardt backpropagation.
        % if (N<=2) % to avoid overfitting
        %     hiddenLayerSize = i;  % setting hidenLayerSize == # of parameters
        % else
        %     hiddenLayerSize = i+1;  % setting hidenLayerSize == # of parameters
        % end
        hiddenLayerSize = i+1;
        %start optimization
        Nepoch =1000;  % train 1000 times
        R2Tmax=0.0;
        
        R2T=zeros(Nepoch);
        % For all data points in each case
        for k=1:Nepoch
            net = fitnet(hiddenLayerSize,trainFcn); % create network object and store into "net"
            % Choose Input and Output Pre/Post-Processing Functions
            % For a list of all processing functions type: help nnprocess
            net.input.processFcns = {'removeconstantrows','mapminmax'};
            net.output.processFcns = {'removeconstantrows','mapminmax'};
            net.divideParam.trainRatio = 75/100;
            net.divideParam.valRatio = 15/100;
            net.divideParam.testRatio = 15/100;
            
            [net, traininfo] = train(net, X, t);  % train function will take first the network to train as
            ytrained = net(X(:,traininfo.trainInd)); %
            yTraintrue = t(traininfo.trainInd);
            
            mdl=fitlm(yTraintrue,ytrained);
            R2T(k)=mdl.Rsquared.Ordinary;
            % pT=mdl.Coefficients.pValue(2);
            % rmse_Train(k) = mdl.RMSE;
            
            if (R2T(k) > R2Tmax)
                R2Tmax = R2T(k);
                netmax=net;
                traininfomax = traininfo;
                mdlmax=mdl;
                % [netmax, traininfomax] = train(netmax, X, t);
            end
        end
        
        % net = fitnet(hiddenLayerSize,trainFcn); % create network object and store into "net"
        % Choose Input and Output Pre/Post-Processing Functions
        % For a list of all processing functions type: help nnprocess
        % % netmax.input.processFcns = {'removeconstantrows','mapminmax'};
        netmax.output.processFcns = {'removeconstantrows','mapminmax'};
        netmax.divideParam.trainRatio = 100/100;
        netmax.divideParam.valRatio = 0/100;
        netmax.divideParam.testRatio = 0/100;
        % % 
        [netmax, traininfomax] = train(netmax, X, t);  % train function will take first the network to train as
        % % %             ytrained = net(X(:,traininfo.trainInd)); %
        % % %             yTraintrue = t(traininfo.trainInd);
        % % %             R2T(k)=mdl.Rsquared.Ordinary;
        ytrainedall = netmax(X(:,traininfomax.trainInd));
        yTraintrueall = t(traininfomax.trainInd);
        
        mdlmax=fitlm(yTraintrueall,ytrainedall);
        
        R2Tall(i)=mdlmax.Rsquared.Ordinary
        pTall(i)=mdlmax.Coefficients.pValue(2);
%         rmse_Trainall(i) = mdlmax.RMSE; %Do not use this. This test the
%         linear fitting model not the Machine learning model
        rmse_Trainall(i) = sqrt((sum((ytrainedall-yTraintrueall).^2)/nr));
       
    end
    Nu=1:N;
    title = {'Number of parameters' 'RMSE' 'R2' 'p' 'P-measured' 'P-predicted'};
    xlswrite (outfilename,title, 'Sheet1', 'A1');
    xlswrite (outfilename,Nu','Sheet1', 'A2');
    xlswrite (outfilename,rmse_Trainall','Sheet1', 'B2');
    xlswrite (outfilename,R2Tall','Sheet1', 'C2');
    xlswrite (outfilename,pTall','Sheet1', 'D2');
    xlswrite (outfilename,t','Sheet1', 'E2');
    xlswrite (outfilename,ytrainedall','Sheet1', 'F2');
end

toc; % End recording time
