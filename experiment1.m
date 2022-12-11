sca;
close all;
clearvars;

%bsusb('init')
PsychDefaultSetup(2);
Screen('Preference', 'SkipSyncTests', 1);

grey = [0.345, 0.345 0.345];

blocks = 1;
trials = 1;

InitializePsychSound;
wv = append(pwd,'\audio\tone.wav');
[y, freq] = psychwavread(wv);
wavedata = y';
wavedata = [wavedata ; wavedata];


[window, windowRect] = PsychImaging('OpenWindow', 0, grey);


for block = 1 : blocks
    single = [zeros(1,trials),ones(1,trials),2*ones(1,trials),3*ones(1,trials)];
    ryhtmic = [4*ones(1,trials),5*ones(1,trials),6*ones(1,trials),7*ones(1,trials)];
    tasks = [single,ryhtmic];
    %tasks = [ryhtmic];
    tasks = tasks(randperm(numel(tasks)));
    
    for prompt = 1 : length(tasks)

        imageTexture = Screen('MakeTexture', window, getPrompt('cross'));
        Screen('DrawTexture', window, imageTexture, [], [], 0);
        Screen('Flip', window);
        WaitSecs(5);

        if tasks(prompt)== 0  
          %  bsusb('send', '', 10)
            showSinglePropmt('pause',grey,window)

        elseif tasks(prompt) == 1
         %   bsusb('send', '', 20)
            showSinglePropmt('left',grey,window)

        elseif tasks(prompt) == 2
          %  bsusb('send', '', 30)
            showSinglePropmt('work',grey,window)

        elseif tasks(prompt) == 3
          %  bsusb('send', '', 40)
            showSinglePropmt('right',grey,window)

        elseif tasks(prompt) == 4
          %  bsusb('send', '', 110)
            showRythmPropmt('r_pinch',window,grey,freq,wavedata)
           
        elseif tasks(prompt) == 5
        %    bsusb('send', '', 120)
            showRythmPropmt('r_stop',window,grey,freq,wavedata)

        elseif tasks(prompt) == 6
         %   bsusb('send', '', 130)
            showRythmPropmt('r_left',window,grey,freq,wavedata)
           
         elseif tasks(prompt) == 7
         %   bsusb('send', '', 140)
            showRythmPropmt('r_right',window,grey,freq,wavedata)
           
        end
        imageTexture = Screen('MakeTexture', window, getPrompt('relax'));
        Screen('DrawTexture', window, imageTexture, [], [], 0);
        Screen('Flip', window);
        if randsample(99,1) > 80
         %   bsusb('send', '', 1)
            WaitSecs(15);
        else 
            WaitSecs(10);
        end

        if mod(prompt, length(tasks)) == 0 
            imageTexture = Screen('MakeTexture', window, getPrompt('rest'));
            Screen('DrawTexture', window, imageTexture, [], [], 0);
            Screen('Flip', window);
            KbWait();
        end

    end
end    
%bsusb('close')
sca;

function image = getPrompt(name)
    location = append(pwd,'\images\experiment prompts\',name,'.png');
    image = imread(location);
end
function playAudio(freq,wavedata,window,grey)
    for stimuli = 1 : 5
        pahandle = PsychPortAudio('Open', [], [], 0, freq, 2);
        PsychPortAudio('FillBuffer', pahandle, wavedata);
        PsychPortAudio('Start', pahandle, 1, 0, 0, 0.1);
        WaitSecs(0.2);
        if stimuli == 1
            Screen('FillRect', window, grey);
            Screen('Flip', window);
        end 
        WaitSecs(0.4);
        PsychPortAudio('Stop', pahandle);
        PsychPortAudio('Close', pahandle);
    end
end
function showRythmPropmt(class,window,grey,freq,wavedata)
    imageTexture = Screen('MakeTexture', window, getPrompt(class));
    Screen('DrawTexture', window, imageTexture, [], [], 0);
    Screen('Flip', window);
    WaitSecs(6);
    imageTexture = Screen('MakeTexture', window, getPrompt('cross'));
    Screen('DrawTexture', window, imageTexture, [], [], 0);
    Screen('Flip', window);
    WaitSecs(6);
    for count = 1 : 2
        imageTexture = Screen('MakeTexture', window, getPrompt('r_cue'));
        Screen('DrawTexture', window, imageTexture, [], [], 0);
        Screen('Flip', window);
        if count < 2
            playAudio(freq,wavedata,window,grey)
        else
            WaitSecs(0.2);
            Screen('FillRect', window, grey);
            Screen('Flip', window);
            WaitSecs(3.8);
        end
    end
end
function showSinglePropmt(class,grey,window)
    imageTexture = Screen('MakeTexture', window, getPrompt(class));
    Screen('DrawTexture', window, imageTexture, [], [], 0);
    Screen('Flip', window);
    WaitSecs(6);

    imageTexture = Screen('MakeTexture', window, getPrompt('cross'));
    Screen('DrawTexture', window, imageTexture, [], [], 0);
    Screen('Flip', window);
    WaitSecs(6);                
    
    imageTexture = Screen('MakeTexture', window, getPrompt('cue'));
    Screen('DrawTexture', window, imageTexture, [], [], 0);
    Screen('Flip', window);
    WaitSecs(0.2);
    Screen('FillRect', window, grey);

    Screen('Flip', window);
    WaitSecs(1.8)
end
function drawVideo(file,window, windowRect,times,width, height)
    for times = 1 : times
        movie = Screen('OpenMovie',window,append('C:\Users\Alberto\Documents\BCI\images\experiment assets\',file,'.mp4'));
        Screen('PlayMovie', movie,1);

        while ~KbCheck
            % Wait for next movie frame, retrieve texture handle to it
            frame = Screen('GetMovieImage', window, movie);
            % Valid texture returned? A negative value means end of movie reached:
            if frame<=0
                break;
            end
            Screen('DrawTexture', window, frame, [], [windowRect(3)/width windowRect(4)/height windowRect(3)- windowRect(3)/width windowRect(4) - windowRect(4)/height]);
            Screen('Flip', window);
        end
    end
end