clear
close all
clc

input_path = 'E:\Document\鱼尾模型\2018年3月6日\after_removed_noise_experiment_data\f40A30\';
input_file_name =                                                                'f040A30_the_3_time.csv';
moving_frequency =                                                                0.40;
output_path = 'E:\Document\鱼尾模型\2018年3月6日\picked_label\';
sample_rate = 20;
extend_period_times = 2;

data = csvread(strcat(input_path, input_file_name));

time = data(:,2)';
angle = data(:,3)';    angle = (angle - min(angle)) / (max(angle) - min(angle)) - 0.5;
thrust = data(:,5)';

figure(1)
hold on
plot(time, angle, 'r')
plot(time, thrust, 'b')
plot([time(1,1), time(1,end)], [0, 0], 'g')

click_control = 1;
point_No = 0;
standard_line = true;
pick_label_data = zeros(1,2);
while(click_control==1)
    
    pos = ginput(1);
    x = pos(1,1);    y = pos(1,2);
    
    if y > max(thrust)
        click_control = 0;
    else
        if standard_line == true
            standard_line = false;
            plot([time(1,1), time(1,end)], [y, y], 'g')
        else

            point_No = point_No + 1;
            
            start_point = round((x - time(1,1)) * sample_rate + 1);
            while(1)
                if (angle(1,start_point)>=0) && (angle(1,start_point-1)<0)
                    break
                else
                    start_point = start_point - 1;
                end
            end
            
            pick_label_data(point_No, :) = [start_point, start_point + round(extend_period_times*sample_rate/moving_frequency) - 1];

            disp(['point_No:',int2str(point_No),' pos:',num2str(pos)])
            plot(time(1, start_point), angle(1, start_point), 'k+')
            
        end
    end
    
end

close all

fp = fopen(strcat(output_path, input_file_name), 'w');
% fprintf(fp,'start,end\n');
for m1 = 1:length(pick_label_data(:,1))
    fprintf(fp, '%d,%d\n', pick_label_data(m1,1), pick_label_data(m1,2));
end
fclose(fp);

disp('saved successfully!')














