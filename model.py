# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 09:28:51 2021

@author: Administrator
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.model_zoo as model_zoo
import numpy as np

''''
model_v3是在model_v1的基础上进行调整，主要是增加了编码器和解码器的最大层，最大层的通道数为16，其他暂且先保持不变
'''

class F_Net(nn.Module):
	def __init__(self):
		super(F_Net,self).__init__()
		
		self.conv_in = nn.Conv2d(9,16,kernel_size=3,stride=1,padding=1,bias=False)		   
		self.conv_16_16 = nn.Conv2d(16,16,kernel_size=3,stride=1,padding=1,bias=False) 
		self.conv_16_16_D3 = nn.Conv2d(16,16,kernel_size=3,stride=1,padding=3,dilation=3,bias=False) 
		self.conv_16_16_D6 = nn.Conv2d(16,16,kernel_size=3,stride=1,padding=6,dilation=6,bias=False)         
		self.conv_16_3 = nn.Conv2d(16,3,kernel_size=3,stride=1,padding=1,bias=False)
        
		self.relu = nn.ReLU(inplace=True)
		self.norm = nn.InstanceNorm2d(16, affine=True)
        
		self.ksup = K_Sup()
		self.esup = E_Sup()
		self.ed = En_Decoder()
        

	def forward(self,x):
        
		kout = self.ksup(x)
		eout = self.esup(x)
                    
		f_out = self.ed(x,kout,eout) + x 
		
		return kout,eout,f_out

class K_Sup(nn.Module):
	def __init__(self):
		super(K_Sup,self).__init__()
		
		self.conv_in = nn.Conv2d(3,16,kernel_size=3,stride=1,padding=1,bias=False)		   
		self.conv_16_16 = nn.Conv2d(16,16,kernel_size=3,stride=1,padding=1,bias=False)	
		self.conv_32_16 = nn.Conv2d(32,16,kernel_size=3,stride=1,padding=1,bias=False)	
		self.conv_48_16 = nn.Conv2d(48,16,kernel_size=3,stride=1,padding=1,bias=False)	        
		self.conv_64_16 = nn.Conv2d(64,16,kernel_size=3,stride=1,padding=1,bias=False)	
		self.conv_16_3 = nn.Conv2d(16,3,kernel_size=3,stride=1,padding=1,bias=False)	
		self.relu = nn.ReLU(inplace=True)
		self.norm = nn.InstanceNorm2d(16, affine=True) 
        

	def forward(self,x):
		x1_1 = self.relu(self.norm(self.conv_in(x)))
		x1_2 = self.relu(self.norm(self.conv_16_16(x1_1)))  
		x1_3 = self.relu(self.norm(self.conv_16_16(x1_2)))  	   
		x1_4 = self.relu(self.norm(self.conv_32_16(torch.cat((x1_2,x1_3),1))))  
		x1_5 = self.relu(self.norm(self.conv_48_16(torch.cat((x1_2,x1_3,x1_4),1))))  
		x1_6 = self.relu(self.norm(self.conv_64_16(torch.cat((x1_2,x1_3,x1_4,x1_5),1))))   
		x_out = self.conv_16_3(x1_6)
		
		return x_out


class E_Sup(nn.Module):
	def __init__(self):
		super(E_Sup,self).__init__()
		
		self.conv_in = nn.Conv2d(3,16,kernel_size=3,stride=1,padding=1,bias=False)
		   
		self.conv_16_16 = nn.Conv2d(16,16,kernel_size=3,stride=1,padding=1,bias=False)
		self.conv_16_16_D2 = nn.Conv2d(16,16,kernel_size=3,stride=1,padding=2,dilation=2,bias=False)
		self.conv_16_16_D4 = nn.Conv2d(16,16,kernel_size=3,stride=1,padding=4,dilation=4,bias=False)        
		self.conv_32_16 = nn.Conv2d(32,16,kernel_size=3,stride=1,padding=1,bias=False)       

		self.conv_16_3 = nn.Conv2d(16,3,kernel_size=3,stride=1,padding=1,bias=False)
		self.relu = nn.ReLU(inplace=True)
		self.norm = nn.InstanceNorm2d(16, affine=True)
        

	def forward(self,x):
		x1_1 = self.relu(self.norm(self.conv_in(x)))
		x1_2 = self.relu(self.norm(self.conv_16_16_D2(x1_1)))  
		x1_3 = self.relu(self.norm(self.conv_32_16(torch.cat((x1_1,x1_2),1))))  	   
		x1_4 = self.relu(self.norm(self.conv_16_16_D4(x1_3))) 
		x1_5 = self.relu(self.norm(self.conv_32_16(torch.cat((x1_3,x1_4),1)))) 
		x1_6 = self.relu(self.norm(self.conv_16_16_D2(x1_5)))
         
		x_out = self.conv_16_3(x1_6)  
		
		return x_out + x 

class En_Decoder(nn.Module):
	def __init__(self):
		super(En_Decoder,self).__init__()
		self.conv_12_32 = nn.Conv2d(9,9,kernel_size=3,stride=1,padding=1,bias=False)       
		self.conv_32_32 = nn.Conv2d(9,9,kernel_size=3,stride=1,padding=1,bias=False)
		self.conv_32_64 = nn.Conv2d(9,18,kernel_size=3,stride=1,padding=1,bias=False)
        
		self.conv_64_64 = nn.Conv2d(18,18,kernel_size=3,stride=1,padding=1,bias=False)	
		self.conv_64_32 = nn.Conv2d(18,9,kernel_size=3,stride=1,padding=1,bias=False)	
		self.conv_64_128 = nn.Conv2d(18,36,kernel_size=3,stride=1,padding=1,bias=False)	
		self.conv_128_128 = nn.Conv2d(36,36,kernel_size=3,stride=1,padding=1,bias=False)
        
		self.conv_128_64 = nn.Conv2d(36,18,kernel_size=3,stride=1,padding=1,bias=False)
		self.conv_32_3 = nn.Conv2d(9,3,kernel_size=3,stride=1,padding=1,bias=False)

		
		self.relu = nn.ReLU(inplace=True)	
		self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

	



	def _upsample_add(self,x,y):
		_,_,H,W = y.size()
		return F.upsample(x,size=(H,W),mode='bilinear') + y

	def _upsample(self,x,y):
		_,_,H,W = y.size()
		return F.upsample(x,size=(H,W),mode='bilinear')

	def forward(self,x,ksup,esup):
        
		xtin = torch.cat((x,ksup,esup),1)   
        
		x1_1 = self.relu(self.conv_12_32(xtin)) 
		x1_2 = self.relu(self.conv_32_32(x1_1))   		
	
		#MaxPool DownSample:128-->64
		x2_in = self.maxpool(x1_2)
		
		#Left Conv2
		x2_1 = self.relu(self.conv_32_64(x2_in))
		x2_2 = self.relu(self.conv_64_64(x2_1))
		x2_3 = self.relu(self.conv_64_64(x2_2))

  
		#MaxPool DownSample:64-->32
		x3_in = self.maxpool(x2_3)
		
		#Left Conv3
		x3_1 = self.relu(self.conv_64_128(x3_in))
		x3_2 = self.relu(self.conv_128_128(x3_1))
		x3_3 = self.relu(self.conv_128_128(x3_2))
		x3_4 = self.relu(self.conv_128_128(x3_3))        
		x3_5 = self.relu(self.conv_128_128(x3_4))

		 
		#Reduce the number of feature maps:256-128	  
		x4_in = self.conv_128_64(x3_5)
		
		#Right Conv2
		#Add high resolution feature map and low resolution feature map and up_sample on low resolution image	   
		x4_in = torch.cat((self._upsample(x4_in,x2_3),x2_3),1) 
		x4_1 = self.relu(self.conv_128_64(x4_in))
		x4_2 = self.relu(self.conv_64_64(x4_1))
		x4_3 = self.relu(self.conv_64_64(x4_2))
		
		#Reduce the number of feature maps:128-64			
		x5_in = self.conv_64_32(x4_3)

		#Right Conv1
		#Add high resolution feature map and low resolution feature map and up_sample on low resolution image		  
		x5_in = torch.cat((self._upsample(x5_in,x1_2),x1_2),1) 
		x5_1 = self.relu(self.conv_64_32(x5_in)) 	
		x5_2 = self.relu(self.conv_32_32(x5_1)) 
        
		x_out = self.conv_32_3(x5_2)	
		
		return x_out