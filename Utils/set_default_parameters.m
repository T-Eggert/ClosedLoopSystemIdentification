% copyright: T. Eggert  (2019)
%            Ludwig-Maximilians Universität, Munich
%            Department of Neurology
%            Fraunhoferstr. 20
%            D-82152 Planegg/Martinsried
%            Germany
%            e-mail: eggert@lrz.uni-muenchen.de
%
%  parameters=set_default_parameters(parameters,default_parameters)
%    Fields of parameters that are not contained in default_parameters are removed,
%    fields of default_parameters that are not contained in parameters are added to
%    the structure parameters.function parameters=set_default_parameters(parameters,default_parameters)
function parameters=set_default_parameters(parameters,default_parameters)
if isempty(parameters),
   parameters=default_parameters;
   return;
end;

N=numel(parameters);
if ~isstruct(default_parameters),
   for i=1:N,
      if isnan(parameters(i)),
         parameters(i)=default_parameters(i);
      end;
   end;
else
   FNames=fieldnames(default_parameters);

   for i=1:N,
      for k=1:length(FNames),
         if ~isfield(parameters(i),FNames{k}),
            new_s(i).(FNames{k})=default_parameters.(FNames{k});
         else
            new_s(i).(FNames{k})=parameters(i).(FNames{k});
         end;
      end;
   end;
   clear parameters
   parameters=new_s;
   
end

