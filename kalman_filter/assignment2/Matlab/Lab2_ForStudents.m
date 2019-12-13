clearvars

MotionModelTot = [{'ConstVelWNAcc'}, {'PiecewiseVelWNAcc'}, {'ConstAccW'}, {'PiecewiseAccW'}];
MotionModel = MotionModelTot{1}; % Chage argument to change MotionModel considered

N = 100; % Number of measurement points
R = 1; % Measurement noise variance
T = 1; % Sample time
var_q = 1; % Change here the value of the process noise variance

switch MotionModel
    
    case 'ConstVelWNAcc' % Constant velocity white noise accelleration
        [x, z, A, Q, G, H] = gen_data13(var_q, R, N);
        B = [0; 0];
        C = [1 0];
        D = 0;
        x_0 = [z(1); (z(2)-z(1))/T];
        P_0 = inv([R R/T; R/T 2*R/T^2]);
        
    case 'PiecewiseVelWNAcc' % Piecewise constant velocity white noise accelleration
        [x, z, A, Q, G, H] = gen_data14(var_q, R, N);
        B = [0; 0];
        C = [1 0];
        D = 0;
        x_0 = [z(1); (z(2)-z(1))/T];
        P_0 = inv([R R/T; R/T 2*R/T^2]);
        
        
    case 'ConstAccW' % Constant accelleration Wiener
        [x, z, A, Q, G, H] = gen_data15(var_q, R,N);
        B = [0; 0; 0];
        C = [1 0 0];
        D = 0;
        x_0 = [z(1); (z(2)-z(1))/T; 0];
        P_0 = [R R/T 2*R/T^2;R/T 2*R/(T^2) 3*R/(T^3);2*R/(T^2) 3*R/(T^3) 0]^(-1);
        
    case 'PiecewiseAccW' % Piecewise constant accelleration Wiener
        [x, z, A, Q, G, H] = gen_data16(var_q, R, N);
        B = [0; 0; 0];
        C = [1 0 0];
        D = 0;
        x_0 = [z(1); (z(2)-z(1))/T; 0];
        P_0 = [R R/T 2*R/T^2;R/T 2*R/(T^2) 3*R/(T^3);2*R/(T^2) 3*R/(T^3) 0]^(-1);
end