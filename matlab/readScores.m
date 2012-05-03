function [scores, labels] = readScores(path, OOS)
%%Path is the path to the file with scores.
%OOS determines how to treat out of set languages, it can be either of:
%  'ignore'  - remove scores from out of set utterances
%  'include' - include out of set utterances normally
%  'ubm'     - create universal background model for out of set language
%              by copying each target utterance as out of set language.
%              It is assumed that there are no out of set utterances
  temp = dlmread(path);
  labels = mod(temp(:, 1), 13)';
  
  colSelection = (1:size(temp, 2) ~= 1);%  & (1:size(temp, 2) ~= 14);
  
  if strcmp(OOS,'ignore')
      rowSelection = labels~=0;
      scores = temp(rowSelection, colSelection)';
      labels = labels(rowSelection);
  elseif strcmp(OOS, 'include')
      scores = temp(:, colSelection)';
      labels(labels==0) = 13;
  else
      labels = cat(2, labels, (ones(size(labels))*13));
      scores = cat(1, temp(:, colSelection), temp(:, colSelection))';
  end

      
  
  %sys1cols = 2:13;%The columns with primary dialects
  %sys2cols = [15:17 5:13];%The columns with secondary dialects
  %cellscores = {temp(:, sys1cols)', temp(:, sys2cols)'};
  %
  %selection = (1:size(temp, 2) ~= 1) & (1:size(temp, 2) ~= 14);
  %superscores = temp(:, selection)';