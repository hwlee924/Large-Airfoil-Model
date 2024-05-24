
% function takes two txt files (upper surface and lower surface)
% reorders and concatenates in correct order

% no cleaning in place yet 
% needs to convert to double 
% write to csv file 

% eventually get script to access and write files directly to OG folder


function cleaned = clean(fn1, fn2)

    orificesca = readcell('AAorifices.txt');

    if fn1(2:end) ~= fn2(2:end) % ensure correct files used
        cleaned = disp('ERROR: DIFFERENT UPPER AND LOWER FILES!');
 
    elseif fn1(1) == 'L' % ensure correct order of files
        fn1 = fn2;
        fn2(1) = 'L'; 
    end 
    % txt files from extracttable

    airstr = 'NASA SC(3)-0712(B)';

    if fn1(3)=='n' || fn1(3)=='m'
        attacksign = 'm';
    elseif fn1(3)=='p'
        attacksign = '';
    end 

    attackstr = [attacksign fn1(4) '.' fn1(5)];
    restr = [fn1(8) '.' fn1(9)];
    machstr = [fn1(11) '.' fn1(12:13)];

    rawstr1 = fileread(fn1);
    ca1 = regexp(rawstr1, '\r\n|\r|\n', 'split'); % converts to cell array with one value per cell
    ca1 = flip(ca1); % flips order for upper surface

    rawstr2 = fileread(fn2); % repeat for lower surface 
    ca2 = regexp(rawstr2, '\r\n|\r|\n', 'split');
    
    ca = [ca1 , ca2]'; % make vertical

    [r,~] = size(ca); % get number of rows

    cadup = cell(r,1); % make duplicate to clean

    for i = 1:r 
        
        str = ca{i}; % take line as string 
        
        % address NaN cases

        if str(1) == '﻿' % delete weird invisible character!!
            str(1) = [];
        end 

        cadup{i} = str2double(str); % convert string to double 
        cadupi = cadup{i};

        %clean up using math filters and sanity checks

        if abs(cadupi) > 10 % address missed decimals
            cadup{i} = cadupi / 10000;

        % contingency if fails all filters so MASTER HINGOO can have a look
        elseif isnan(cadupi)
            cadup{i} = ca{i};
        end 
        
    end 
    mach = str2double(machstr); % add mach number

    % add column to new / existing csv

    newfn = [airstr '_A' attackstr '_A_Re' restr 'e6.csv']; % 'NASA SC(3)-0712(B)_A-4.0_A_Re7.0e6'

    if exist(newfn,"file") == 0 % if file does not exist

        cleaned = [{[],mach};orificesca , cadup];
        newfn = ['NEW CHECK ' newfn]; % indicate new file in name
        writecell(cleaned,newfn)   % write to new csv with correct name format
        movefile (newfn, 'CHECK CSV') % move to check folder

    elseif exist(newfn,"file") == 2

        oldcell = readcell(newfn); % read existing file
        oldcell(1,1) = {[]};
        cadup = [{mach} ; cadup]; % concat mach number to new column
        cleaned = [oldcell , cadup]; % display full updated cell array

        newfn = ['UPDATED CHECK ' newfn] % indicate updated file in name
        writecell(cleaned,newfn)   % write to new csv with correct name format
        movefile (newfn, 'CHECK CSV') % move to check folder


    end 

    

end 

