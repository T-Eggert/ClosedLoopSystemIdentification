function indices=strmatch_array(pattern_list,search_list,force_error_on_NotFound)
if nargin<3,
   force_error_on_NotFound=[];
end;
if isempty(force_error_on_NotFound),
   force_error_on_NotFound=true;
end;

if iscell(pattern_list),
   N=length(pattern_list);
else
   N=size(pattern_list,1);
end;

if iscell(search_list),
   for k=1:length(search_list),
      search_list{k}=strtrim(search_list{k});
   end;
else
   tmp=[];
   for k=1:size(search_list,1),
      tmp=strvcat(tmp,strtrim(search_list(k,:)));
   end;
   search_list=tmp;
   clear tmp
end;

indices=NaN(N,1);
for k=1:N,
   if iscell(pattern_list),
      s=pattern_list{k};
   else
      s=strtrim(pattern_list(k,:));
   end;
   i=strmatch(s,search_list,'exact');
   if ~isempty(i),
      indices(k)=i(1);
   elseif force_error_on_NotFound,
      error(sprintf('pattern %s not found in search list!',s));
   end;
end;

end
