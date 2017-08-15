function parseRosbag(filename)
    % Get rosbag
    bag = rosbag(strcat('..\rosbags\',filename,'.bag'));
    bagSelect = select(bag,'Topic','/total_message');
    bagMsgs = readMessages(bagSelect);
    
    %Return early if no total messages in bag
    if(isempty(bagMsgs))
        disp(strcat(['No total messages in ',filename]));
        return
    end
    
    %Get timestamps
    for msgNum = 1:size(bagMsgs,1)
        msg = bagMsgs{msgNum};
        vars = {'Time'};
        timeArray(msgNum) = double(msg.Header.Stamp.Nsec)*1e-9 + ...
            double(msg.Header.Stamp.Sec) - bag.StartTime;
    end    
    bagTable = array2table(timeArray','VariableNames',vars);
    
    %Get integrated message definition
    msgInfo = rosmsg('show','morpheus_skates/integrated_message');
    msgInfoCellArray = strread(msgInfo, '%s', 'delimiter', sprintf('\n'));
    
    %Iterate through integrated message fields
    for fieldNum = 2:size(msgInfoCellArray,1)
        fieldString = strsplit(msgInfoCellArray{fieldNum});
        if(isempty(fieldString{1}))
            continue
        end
        
        %Clear common variables
        clear topicTable vars topic
        
        %Check for known message types to recursively decompose
        if(strcmp(fieldString{2},'LeftFeedback') || ...
                strcmp(fieldString{2},'RightFeedback') || ...
                strcmp(fieldString{2},'LeftCommand') || ...
                strcmp(fieldString{2},'RightCommand') || ...
                strcmp(fieldString{2},'NormalizedForce') || ...
                strcmp(fieldString{2},'PoundsValues'))
            topicTable = parseTopic(bagMsgs,fieldString{2},...
                strcat('morpheus_skates/',fieldString{1}));  
        end
        
        if(strcmp(fieldString{2},'CentreOfMassKinect'))
            for msgNum = 1:size(bagMsgs,1)
                msg = bagMsgs{msgNum};
                vars{1} = 'CentreOfMassKinect_X';
                vars{2} = 'CentreOfMassKinect_Y';
                vars{3} = 'CentreOfMassKinect_Z';
                topic(msgNum,:) = ...
                    double(eval('msg.CentreOfMassKinect'))';
            end
            topicTable = array2table(topic,'VariableNames',vars);            
        end
        
        if(strcmp(fieldString{2},'HipLeft'))
            for msgNum = 1:size(bagMsgs,1)
                msg = bagMsgs{msgNum};
                vars{1} = 'HipLeft_X';
                vars{2} = 'HipLeft_Y';
                vars{3} = 'HipLeft_Z';
                topic(msgNum,:) = ...
                    double(eval('msg.HipLeft'))';
            end
            topicTable = array2table(topic,'VariableNames',vars);             
        end
        
        if(strcmp(fieldString{2},'HipRight'))
            for msgNum = 1:size(bagMsgs,1)
                msg = bagMsgs{msgNum};
                vars{1} = 'HipRight_X';
                vars{2} = 'HipRight_Y';
                vars{3} = 'HipRight_Z';
                topic(msgNum,:) = ...
                    double(eval('msg.HipRight'))';
            end
            topicTable = array2table(topic,'VariableNames',vars);             
        end
        
        if(strcmp(fieldString{2},'FootLeft'))
            for msgNum = 1:size(bagMsgs,1)
                msg = bagMsgs{msgNum};
                vars{1} = 'FootLeft_X';
                vars{2} = 'FootLeft_Y';
                vars{3} = 'FootLeft_Z';
                topic(msgNum,:) = ...
                    double(eval('msg.FootLeft'))';
            end
            topicTable = array2table(topic,'VariableNames',vars);            
        end

        if(strcmp(fieldString{2},'FootRight'))
            for msgNum = 1:size(bagMsgs,1)
                msg = bagMsgs{msgNum};
                vars{1} = 'FootRight_X';
                vars{2} = 'FootRight_Y';
                vars{3} = 'FootRight_Z';
                topic(msgNum,:) = ...
                    double(eval('msg.FootRight'))';
            end
            topicTable = array2table(topic,'VariableNames',vars);            
        end
        
        if(strcmp(fieldString{2},'UserPositionOffset'))
            for msgNum = 1:size(bagMsgs,1)
                msg = bagMsgs{msgNum};
                vars{1} = 'UserPositionOffset';
                topic(msgNum,1) = ...
                    double(eval('msg.UserPositionOffset'));
            end
            topicTable = array2table(topic,'VariableNames',vars);
        end
        
        %Concatenate table entries - try/catch handles skipped fields
        try
            bagTable = [bagTable topicTable];
        catch
            continue
        end
    end
    
    %Write tables to .mat and .csv files
    save(strcat('..\Mat Files\',filename,'.mat'),'bagTable'); 
    writetable(bagTable,strcat('..\Mat Files\',filename,'.csv'));
end


function topicTable = parseTopic(bagMsgs,messageName,messageType) 
    msgInfo = rosmsg('show',messageType);
    msgInfoCellArray = strread(msgInfo, '%s', 'delimiter', sprintf('\n'));
    
    for msgNum = 1:size(bagMsgs,1)
        msg = bagMsgs{msgNum};
        for fieldNum = 2:size(msgInfoCellArray,1)
            fieldString = strsplit(msgInfoCellArray{fieldNum});
            if(isempty(fieldString{1}))
                continue
            end
            fieldName = fieldString{2};
            vars{fieldNum-1} = strcat(messageName,'_',fieldName);
            topic(msgNum,fieldNum-1) = ...
                double(eval(strcat('msg.',messageName,'.',fieldName)));
        end
    end
    topicTable = array2table(topic,'VariableNames',vars);
end