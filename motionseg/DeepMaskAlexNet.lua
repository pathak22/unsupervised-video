--[[ DeepMask model:
When initialized, it creates/load the common trunk, the maskBranch and the
scoreBranch or the colorBranch or the flowBranch.
---- deepmask class members:
-- self.trunk: the common trunk (pre-trained resnet50)
-- self.maskBranch: the mask head architecture
-- self.scoreBranch: the score head architecture
-- self.colorBranch: the colorization head architecture
-- self.flowBranch: the flow head architecture
]]

require 'nn'
require 'nnx'
require 'cunn'
require 'cudnn'
local utils = paths.dofile('utilsModel.lua')
paths.dofile('SpatialSymmetricPadding.lua')

local DeepMask,_ = torch.class('nn.DeepMask','nn.Container')

-- function: conv2linear
local function linear2conv(x)
  if torch.typename(x):find('Linear') then
    -- hard-coding for fc6 and fc7: kSz=kernelSize=inputFeatureMapSize
    local kSz = x.weight:size(2) > 5000 and 6 or 1

    local nInp = x.weight:size(2)/(kSz*kSz)
    local nOut = x.weight:size(1)
    local w = torch.reshape(x.weight,nOut,nInp,kSz,kSz)
    local y = cudnn.SpatialConvolution(nInp,nOut,kSz,kSz,1,1,0,0)
    y.weight:copy(w)
    y.gradWeight:copy(w)
    if x.bias~=nil then
      y.bias:copy(x.bias)
      y.gradBias:copy(x.gradBias)
    end
    return y
  elseif torch.typename(x):find('cudnn.BatchNormalization') then
     x.nDim = 4
     return x
  else
    return x
  end
end

--------------------------------------------------------------------------------
-- function: constructor
function DeepMask:__init(config)
   self.color = config.color
   self.flow = config.flow
   if config.noFC then
      print('| create AlexNet (w/o FCs) Trunk')
   else
      print('| create AlexNet (including FCs) Trunk')
   end
   if config.symmPad then
      print('| using symmetric padding')
   else
      print('| no symmetric padding')
   end
   if config.centralCrop then
      print('| using central cropping')
   else
      print('| no central cropping')
   end
   if config.bottleneck then
      print('| using bottleneck')
   else
      print('| no bottleneck')
   end

   -- create common trunk
   self:createTrunk(config)
   local npt = 0
   local p1  = self.trunk:parameters()
   for k,v in pairs(p1) do npt = npt+v:nElement() end
   print(string.format('| number of paramaters trunk: %d', npt))

   if self.flow then
         -- create flow head
         self:createFlowBranch(config)

         local p5, npf = self.flowBranch:parameters(), 0
         for k,v in pairs(p5) do npf = npf+v:nElement() end
         print(string.format('| number of paramaters flow branch: %d', npf))
         print(string.format('| number of paramaters total: %d', npt+npf))
         return
   end

   -- create mask head
   self:createMaskBranch(config)
   local npm = 0
   local p2  = self.maskBranch:parameters()
   for k,v in pairs(p2) do npm = npm+v:nElement() end
   print(string.format('| number of paramaters mask branch: %d', npm))

   if self.color then
      -- create colorization head
      self:createColorBranch(config)

      local p4, npc = self.colorBranch:parameters(), 0
      for k,v in pairs(p4) do npc = npc+v:nElement() end
      print(string.format('| number of paramaters color branch: %d', npc))
      print(string.format('| number of paramaters total: %d', npt+npm+npc))
   else
      -- create score head
      self:createScoreBranch(config)

      local p3, nps = self.scoreBranch:parameters(), 0
      for k,v in pairs(p3) do nps = nps+v:nElement() end
      print(string.format('| number of paramaters score branch: %d', nps))
      print(string.format('| number of paramaters total: %d', npt+nps+npm))
   end
end

--------------------------------------------------------------------------------
-- function: create common trunk
function DeepMask:createTrunk(config)
   -- size of feature maps at end of trunk
   if config.padAlexNet then
      if config.iSz==180 then
         -- self.fSz = config.noFC and 12 or 5  -- alexnet_padded w/o dilation
         self.fSz = 12  -- alexnet_padded w/ dilation
      else
         print('Unknown size setting !! Cant create AlexNet trunk')
         os.exit()
      end
   else
      -- iSz=227 ; for w/ FC
      -- iSz=179 ; for w/o FC
      if config.iSz==160 then
         self.fSz = config.noFC and 8 or -1
      elseif config.iSz==179 then
         self.fSz = config.noFC and 10 or -1
      elseif config.iSz==227 then
         self.fSz = config.noFC and 13 or 1
      else
         print('Unknown size setting !! Cant create AlexNet trunk')
         os.exit()
      end
   end
   self.channels = config.noFC and 128 or 4096
   self.bottleneck = self.channels*self.fSz*self.fSz

   -- load trunk
   local trunk
   print('    | creating trunk:')
   if #config.useImagenet > 0 then
     print(string.format('    | using Imagenet pre-trained AlexNet: %s',
        config.useImagenet))
     trunk = torch.load(config.useImagenet)
     -- Format of sgross's old fb.resnet training code
     if trunk.state ~= nil then
       trunk = trunk.state.network
     end
     -- remove DataParallelTable
     if torch.type(trunk) == 'nn.DataParallelTable' then
       trunk = trunk:get(1)
     end
     if config.useBN then
       print('    | keeping BatchNorm in pre-trained model (if present)')
     else
       print('    | fixing BatchNorm in pre-trained model (if present)')
       utils.BNtoFixed(trunk, true)
     end
   elseif config.useBN then
     print('    | using AlexNet with BatchNorm from scratch !')
     local alexnet = paths.dofile('./models/alexnetbn.lua')
     trunk = alexnet()
   else
     print('    | using AlexNet without BatchNorm from scratch !')
     local alexnet = config.padAlexNet and paths.dofile(
         './models/alexnet_padded.lua') or paths.dofile('./models/alexnet.lua')
     trunk = alexnet()
   end
  --  print('    | loaded trunk model:')
  --  print(trunk)

   -- remove fc8
   trunk:remove();

   if config.noFC then
      -- remove fc7
      trunk:remove();trunk:remove();
      if torch.typename(trunk.modules[#trunk.modules]):find('BatchNorm') then
        trunk:remove();
      end
      trunk:remove();
      -- remove fc6
      trunk:remove();trunk:remove();
      if torch.typename(trunk.modules[#trunk.modules]):find('BatchNorm') then
        trunk:remove();
      end
      trunk:remove();
      if torch.typename(trunk.modules[#trunk.modules]):find('View') then
        trunk:remove();
      end
      -- remove pool5
      trunk:remove();

      -- crop central pad : see DataSamplerCoco.wSz
      if config.centralCrop then
         trunk:add(nn.SpatialZeroPadding(-1,-1,-1,-1))
      end

      -- add common extra layers
      trunk:add(cudnn.SpatialConvolution(256,128,1,1,1,1))
      if config.useBN then
        trunk:add(cudnn.SpatialBatchNormalization(128))
      end
      trunk:add(nn.ReLU(true))
   else
      if #config.useImagenet > 0 then
        print('    | FC to Conv conversion in pre-trained model')
        local startFCLayer = 16
        if config.useBN then
          startFCLayer = 19
        end
        local j=startFCLayer
        for i=startFCLayer,#trunk.modules do
          if not torch.typename(trunk.modules[i]):find('View') then
           trunk.modules[j] = linear2conv(trunk.modules[i])
           j=j+1
          end
        end
        for j=j,#trunk.modules do
          trunk:remove()
        end
      end

      -- crop central pad : see DataSamplerCoco.wSz
      if config.centralCrop then
         trunk:add(nn.SpatialZeroPadding(-1,-1,-1,-1))
      end
   end
   -- trunk:add(nn.View(config.batch,self.bottleneck))

   -- low-rank bottleneck
   if config.bottleneck then
      trunk:add(nn.Linear(self.bottleneck,512))
      if config.useBN then
         trunk:add(cudnn.BatchNormalization(512))
      end
      self.bottleneck = 512
    end

   -- mirrorPadding
   if config.symmPad then
      utils.updatePadding(trunk, nn.SpatialSymmetricPadding)
   end

   self.trunk = trunk:cuda()

   print('    | finalized trunk model:')
   print(trunk)
   return trunk
end

--------------------------------------------------------------------------------
-- function: create mask branch
function DeepMask:createMaskBranch(config)
   local maskBranch = nn.Sequential()

   -- maskBranch
   if not config.bottleneck then
      maskBranch:add(nn.View(config.batch,self.bottleneck))
   end
   maskBranch:add(nn.Linear(self.bottleneck,config.oSz*config.oSz))
   self.maskBranch = nn.Sequential():add(maskBranch:cuda())

   -- upsampling layer
   if config.gSz > config.oSz then
      local upSample = nn.Sequential()
      upSample:add(nn.Copy('torch.CudaTensor','torch.FloatTensor'))
      upSample:add(nn.View(config.batch,config.oSz,config.oSz))
      upSample:add(nn.SpatialReSamplingEx{owidth=config.gSz,oheight=config.gSz,
         mode='bilinear'})
      upSample:add(nn.View(config.batch,config.gSz*config.gSz))
      upSample:add(nn.Copy('torch.FloatTensor','torch.CudaTensor'))
      self.maskBranch:add(upSample)
   end

   print('    | finalized mask model:')
   print(self.maskBranch)
   return self.maskBranch
end

--------------------------------------------------------------------------------
-- function: create score branch
function DeepMask:createScoreBranch(config)
   local scoreBranch = nn.Sequential()
   if not config.bottleneck then
      scoreBranch:add(nn.View(config.batch,self.bottleneck))
   end
   scoreBranch:add(nn.Dropout(.5))
   scoreBranch:add(nn.Linear(self.bottleneck,1024))
   if config.useBN then
     scoreBranch:add(cudnn.BatchNormalization(1024))
   end
   scoreBranch:add(nn.Threshold(0, 1e-6))

   scoreBranch:add(nn.Dropout(.5))
   scoreBranch:add(nn.Linear(1024,1))

   self.scoreBranch = scoreBranch:cuda()
   print('    | finalized score model:')
   print(self.scoreBranch)
   return self.scoreBranch
end

--------------------------------------------------------------------------------
-- function: create colorization branch
function DeepMask:createColorBranch(config)
   if config.bottleneck then
      print('config.bottleneck in trunk is not supported with Color Task !!')
      os.exit()
   end
   local colorBranch = nn.Sequential()
   colorBranch:add(nn.SpatialFullConvolution(self.channels,256,4,4,2,2,1,1))
   colorBranch:add(nn.ReLU(true))
   colorBranch:add(cudnn.SpatialConvolution(256,313,3,3,1,1,1,1))
   colorBranch:add(nn.SpatialUpSamplingBilinear({oheight=config.cgSz,
                                                   owidth=config.cgSz}))
   self.colorBranch = colorBranch:cuda()
   print('    | finalized color model:')
   print(self.colorBranch)
   return self.colorBranch
end

--------------------------------------------------------------------------------
-- function: create flow branch
function DeepMask:createFlowBranch(config)
   if config.bottleneck then
      print('config.bottleneck in trunk is not supported with Flow Task !!')
      os.exit()
   end
   local flowBranch = nn.Sequential()
   flowBranch:add(cudnn.SpatialConvolution(self.channels,
                                             config.numCl,3,3,1,1,1,1))
   -- upsample if fgSz > 12 (e.g. 100)
   -- flowBranch:add(nn.SpatialUpSamplingBilinear({oheight=config.fgSz,
   --                                                 owidth=config.fgSz}))
   self.flowBranch = flowBranch:cuda()
   print('    | finalized flow model:')
   print(self.flowBranch)
   return self.flowBranch
end

--------------------------------------------------------------------------------
-- function: training
function DeepMask:training()
   self.trunk:training()
   if self.flow then
      self.flowBranch:training()
      return
   end
   self.maskBranch:training()
   if self.color then
      self.colorBranch:training()
   else
      self.scoreBranch:training()
   end
end

--------------------------------------------------------------------------------
-- function: evaluate
function DeepMask:evaluate()
   self.trunk:evaluate()
   if self.flow then
      self.flowBranch:evaluate()
      return
   end
   self.maskBranch:evaluate()
   if self.color then
      self.colorBranch:evaluate()
   else
      self.scoreBranch:evaluate()
   end
end

--------------------------------------------------------------------------------
-- function: to cuda
function DeepMask:cuda()
   self.trunk:cuda()
   if self.flow then
      self.flowBranch:cuda()
      return
   end
   self.maskBranch:cuda()
   if self.color then
      self.colorBranch:cuda()
   else
      self.scoreBranch:cuda()
   end
end

--------------------------------------------------------------------------------
-- function: to float
function DeepMask:float()
   self.trunk:float()
   if self.flow then
      self.flowBranch:float()
      return
   end
   self.maskBranch:float()
   if self.color then
      self.colorBranch:float()
   else
      self.scoreBranch:float()
   end
end

--------------------------------------------------------------------------------
-- function: inference (used for full scene inference)
function DeepMask:inference()
   self:cuda()
   utils.linear2convTrunk(self.trunk,self.fSz)
   self.trunk:evaluate()
   self.trunk:forward(torch.CudaTensor(1,3,800,800))
   if self.flow then
      utils.linear2convHead(self.flowBranch)
      self.flowBranch:evaluate()
      self.flowBranch:forward(torch.CudaTensor(1,512,300,300))
      return
   end

   utils.linear2convHead(self.maskBranch.modules[1])
   self.maskBranch = self.maskBranch.modules[1]
   self.maskBranch:evaluate()
   self.maskBranch:forward(torch.CudaTensor(1,512,300,300))

   if self.color then
      utils.linear2convHead(self.colorBranch)
      self.colorBranch:evaluate()
      self.colorBranch:forward(torch.CudaTensor(1,512,300,300))
   else
      utils.linear2convHead(self.scoreBranch)
      self.scoreBranch:evaluate()
      self.scoreBranch:forward(torch.CudaTensor(1,512,300,300))
   end
end

--------------------------------------------------------------------------------
-- function: clone
function DeepMask:clone(...)
   local f = torch.MemoryFile("rw"):binary()
   f:writeObject(self)
   f:seek(1)
   local clone = f:readObject()
   f:close()

   if select('#',...) > 0 then
      clone.trunk:share(self.trunk,...)
      if self.flow then
         clone.flowBranch:share(self.flowBranch,...)
         return clone
      end
      clone.maskBranch:share(self.maskBranch,...)
      if self.color then
         clone.colorBranch:share(self.colorBranch,...)
      else
         clone.scoreBranch:share(self.scoreBranch,...)
      end
   end

   return clone
end

return DeepMask
