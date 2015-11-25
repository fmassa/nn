local Threshold, parent = torch.class('THNN.Threshold','nn.Module',THNN)

function Threshold:__init(th,v,ip)
   parent.__init(self)
   self.threshold = th or 1e-6
   self.val = v or 0
   if (th and type(th) ~= 'number') or (v and type(v) ~= 'number') then
      error('nn.Threshold(threshold, value)')
   end
   -- default for inplace is false
   self.inplace = ip or false
   if (ip and type(ip) ~= 'boolean') then
      error('in-place flag must be boolean')
   end
   self:validateParameters()
end

function Threshold:updateOutput(input)
   self:validateParameters()
   THNN.errcheck(
      'THNN_RealThreshold_updateOutput',
      input:type(),
      THNN.NULL,
      input:cdata(),
      self.output:cdata(),
      self.val,
      self.threshold,
      self.inplace
   )
   return self.output
end

function Threshold:updateGradInput(input, gradOutput)
   self:validateParameters()
   THNN.errcheck(
      'THNN_RealThreshold_updateGradInput',
      input:type(),
      THNN.NULL,
      input:cdata(),
      gradOutput:cdata(),
      self.gradInput:cdata(),
      self.threshold,
      self.inplace
   )
   return self.gradInput
end

function Threshold:validateParameters()
   self.inplace = self.inplace or false -- backwards compatibility pre inplace
   if self.inplace then
      if self.val > self.threshold then
         error('in-place processing requires value (' .. self.val ..
                  ') not exceed threshold (' .. self.threshold .. ')')
      end
   end
end
