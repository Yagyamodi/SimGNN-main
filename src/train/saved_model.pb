??(
??
D
AddV2
x"T
y"T
z"T"
Ttype:
2	??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
l
BatchMatMulV2
x"T
y"T
output"T"
Ttype:
2		"
adj_xbool( "
adj_ybool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
=
Greater
x"T
y"T
z
"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
?
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
?
Mul
x"T
y"T
z"T"
Ttype:
2	?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
-
Tanh
x"T
y"T"
Ttype:

2
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
P
Unpack

value"T
output"T*num"
numint("	
Ttype"
axisint 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.6.02v2.6.0-rc2-32-g919f693420e8϶&
?
graph_conv_36/graph_conv_36_WVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*.
shared_namegraph_conv_36/graph_conv_36_W
?
1graph_conv_36/graph_conv_36_W/Read/ReadVariableOpReadVariableOpgraph_conv_36/graph_conv_36_W*
_output_shapes

:@*
dtype0
?
graph_conv_36/graph_conv_36_bVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*.
shared_namegraph_conv_36/graph_conv_36_b
?
1graph_conv_36/graph_conv_36_b/Read/ReadVariableOpReadVariableOpgraph_conv_36/graph_conv_36_b*
_output_shapes
:@*
dtype0
?
graph_conv_37/graph_conv_37_WVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *.
shared_namegraph_conv_37/graph_conv_37_W
?
1graph_conv_37/graph_conv_37_W/Read/ReadVariableOpReadVariableOpgraph_conv_37/graph_conv_37_W*
_output_shapes

:@ *
dtype0
?
graph_conv_37/graph_conv_37_bVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_namegraph_conv_37/graph_conv_37_b
?
1graph_conv_37/graph_conv_37_b/Read/ReadVariableOpReadVariableOpgraph_conv_37/graph_conv_37_b*
_output_shapes
: *
dtype0
?
graph_conv_38/graph_conv_38_WVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *.
shared_namegraph_conv_38/graph_conv_38_W
?
1graph_conv_38/graph_conv_38_W/Read/ReadVariableOpReadVariableOpgraph_conv_38/graph_conv_38_W*
_output_shapes

: *
dtype0
?
graph_conv_38/graph_conv_38_bVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_namegraph_conv_38/graph_conv_38_b
?
1graph_conv_38/graph_conv_38_b/Read/ReadVariableOpReadVariableOpgraph_conv_38/graph_conv_38_b*
_output_shapes
:*
dtype0
?
attention_12/att_weightVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameattention_12/att_weight
?
+attention_12/att_weight/Read/ReadVariableOpReadVariableOpattention_12/att_weight*
_output_shapes

:*
dtype0
?
neural_tensor_layer_12/VariableVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!neural_tensor_layer_12/Variable
?
3neural_tensor_layer_12/Variable/Read/ReadVariableOpReadVariableOpneural_tensor_layer_12/Variable*"
_output_shapes
:*
dtype0
?
!neural_tensor_layer_12/Variable_1VarHandleOp*
_output_shapes
: *
dtype0*
shape
: *2
shared_name#!neural_tensor_layer_12/Variable_1
?
5neural_tensor_layer_12/Variable_1/Read/ReadVariableOpReadVariableOp!neural_tensor_layer_12/Variable_1*
_output_shapes

: *
dtype0
?
!neural_tensor_layer_12/Variable_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!neural_tensor_layer_12/Variable_2
?
5neural_tensor_layer_12/Variable_2/Read/ReadVariableOpReadVariableOp!neural_tensor_layer_12/Variable_2*
_output_shapes
:*
dtype0
z
dense_48/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_48/kernel
s
#dense_48/kernel/Read/ReadVariableOpReadVariableOpdense_48/kernel*
_output_shapes

:*
dtype0
r
dense_48/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_48/bias
k
!dense_48/bias/Read/ReadVariableOpReadVariableOpdense_48/bias*
_output_shapes
:*
dtype0
z
dense_49/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_49/kernel
s
#dense_49/kernel/Read/ReadVariableOpReadVariableOpdense_49/kernel*
_output_shapes

:*
dtype0
r
dense_49/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_49/bias
k
!dense_49/bias/Read/ReadVariableOpReadVariableOpdense_49/bias*
_output_shapes
:*
dtype0
z
dense_50/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_50/kernel
s
#dense_50/kernel/Read/ReadVariableOpReadVariableOpdense_50/kernel*
_output_shapes

:*
dtype0
r
dense_50/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_50/bias
k
!dense_50/bias/Read/ReadVariableOpReadVariableOpdense_50/bias*
_output_shapes
:*
dtype0
z
dense_51/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_51/kernel
s
#dense_51/kernel/Read/ReadVariableOpReadVariableOpdense_51/kernel*
_output_shapes

:*
dtype0
r
dense_51/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_51/bias
k
!dense_51/bias/Read/ReadVariableOpReadVariableOpdense_51/bias*
_output_shapes
:*
dtype0
n
Adadelta/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_nameAdadelta/iter
g
!Adadelta/iter/Read/ReadVariableOpReadVariableOpAdadelta/iter*
_output_shapes
: *
dtype0	
p
Adadelta/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdadelta/decay
i
"Adadelta/decay/Read/ReadVariableOpReadVariableOpAdadelta/decay*
_output_shapes
: *
dtype0
?
Adadelta/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdadelta/learning_rate
y
*Adadelta/learning_rate/Read/ReadVariableOpReadVariableOpAdadelta/learning_rate*
_output_shapes
: *
dtype0
l
Adadelta/rhoVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdadelta/rho
e
 Adadelta/rho/Read/ReadVariableOpReadVariableOpAdadelta/rho*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
?
1Adadelta/graph_conv_36/graph_conv_36_W/accum_gradVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*B
shared_name31Adadelta/graph_conv_36/graph_conv_36_W/accum_grad
?
EAdadelta/graph_conv_36/graph_conv_36_W/accum_grad/Read/ReadVariableOpReadVariableOp1Adadelta/graph_conv_36/graph_conv_36_W/accum_grad*
_output_shapes

:@*
dtype0
?
1Adadelta/graph_conv_36/graph_conv_36_b/accum_gradVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*B
shared_name31Adadelta/graph_conv_36/graph_conv_36_b/accum_grad
?
EAdadelta/graph_conv_36/graph_conv_36_b/accum_grad/Read/ReadVariableOpReadVariableOp1Adadelta/graph_conv_36/graph_conv_36_b/accum_grad*
_output_shapes
:@*
dtype0
?
1Adadelta/graph_conv_37/graph_conv_37_W/accum_gradVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *B
shared_name31Adadelta/graph_conv_37/graph_conv_37_W/accum_grad
?
EAdadelta/graph_conv_37/graph_conv_37_W/accum_grad/Read/ReadVariableOpReadVariableOp1Adadelta/graph_conv_37/graph_conv_37_W/accum_grad*
_output_shapes

:@ *
dtype0
?
1Adadelta/graph_conv_37/graph_conv_37_b/accum_gradVarHandleOp*
_output_shapes
: *
dtype0*
shape: *B
shared_name31Adadelta/graph_conv_37/graph_conv_37_b/accum_grad
?
EAdadelta/graph_conv_37/graph_conv_37_b/accum_grad/Read/ReadVariableOpReadVariableOp1Adadelta/graph_conv_37/graph_conv_37_b/accum_grad*
_output_shapes
: *
dtype0
?
1Adadelta/graph_conv_38/graph_conv_38_W/accum_gradVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *B
shared_name31Adadelta/graph_conv_38/graph_conv_38_W/accum_grad
?
EAdadelta/graph_conv_38/graph_conv_38_W/accum_grad/Read/ReadVariableOpReadVariableOp1Adadelta/graph_conv_38/graph_conv_38_W/accum_grad*
_output_shapes

: *
dtype0
?
1Adadelta/graph_conv_38/graph_conv_38_b/accum_gradVarHandleOp*
_output_shapes
: *
dtype0*
shape:*B
shared_name31Adadelta/graph_conv_38/graph_conv_38_b/accum_grad
?
EAdadelta/graph_conv_38/graph_conv_38_b/accum_grad/Read/ReadVariableOpReadVariableOp1Adadelta/graph_conv_38/graph_conv_38_b/accum_grad*
_output_shapes
:*
dtype0
?
+Adadelta/attention_12/att_weight/accum_gradVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*<
shared_name-+Adadelta/attention_12/att_weight/accum_grad
?
?Adadelta/attention_12/att_weight/accum_grad/Read/ReadVariableOpReadVariableOp+Adadelta/attention_12/att_weight/accum_grad*
_output_shapes

:*
dtype0
?
3Adadelta/neural_tensor_layer_12/Variable/accum_gradVarHandleOp*
_output_shapes
: *
dtype0*
shape:*D
shared_name53Adadelta/neural_tensor_layer_12/Variable/accum_grad
?
GAdadelta/neural_tensor_layer_12/Variable/accum_grad/Read/ReadVariableOpReadVariableOp3Adadelta/neural_tensor_layer_12/Variable/accum_grad*"
_output_shapes
:*
dtype0
?
5Adadelta/neural_tensor_layer_12/Variable/accum_grad_1VarHandleOp*
_output_shapes
: *
dtype0*
shape
: *F
shared_name75Adadelta/neural_tensor_layer_12/Variable/accum_grad_1
?
IAdadelta/neural_tensor_layer_12/Variable/accum_grad_1/Read/ReadVariableOpReadVariableOp5Adadelta/neural_tensor_layer_12/Variable/accum_grad_1*
_output_shapes

: *
dtype0
?
5Adadelta/neural_tensor_layer_12/Variable/accum_grad_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:*F
shared_name75Adadelta/neural_tensor_layer_12/Variable/accum_grad_2
?
IAdadelta/neural_tensor_layer_12/Variable/accum_grad_2/Read/ReadVariableOpReadVariableOp5Adadelta/neural_tensor_layer_12/Variable/accum_grad_2*
_output_shapes
:*
dtype0
?
#Adadelta/dense_48/kernel/accum_gradVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*4
shared_name%#Adadelta/dense_48/kernel/accum_grad
?
7Adadelta/dense_48/kernel/accum_grad/Read/ReadVariableOpReadVariableOp#Adadelta/dense_48/kernel/accum_grad*
_output_shapes

:*
dtype0
?
!Adadelta/dense_48/bias/accum_gradVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adadelta/dense_48/bias/accum_grad
?
5Adadelta/dense_48/bias/accum_grad/Read/ReadVariableOpReadVariableOp!Adadelta/dense_48/bias/accum_grad*
_output_shapes
:*
dtype0
?
#Adadelta/dense_49/kernel/accum_gradVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*4
shared_name%#Adadelta/dense_49/kernel/accum_grad
?
7Adadelta/dense_49/kernel/accum_grad/Read/ReadVariableOpReadVariableOp#Adadelta/dense_49/kernel/accum_grad*
_output_shapes

:*
dtype0
?
!Adadelta/dense_49/bias/accum_gradVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adadelta/dense_49/bias/accum_grad
?
5Adadelta/dense_49/bias/accum_grad/Read/ReadVariableOpReadVariableOp!Adadelta/dense_49/bias/accum_grad*
_output_shapes
:*
dtype0
?
#Adadelta/dense_50/kernel/accum_gradVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*4
shared_name%#Adadelta/dense_50/kernel/accum_grad
?
7Adadelta/dense_50/kernel/accum_grad/Read/ReadVariableOpReadVariableOp#Adadelta/dense_50/kernel/accum_grad*
_output_shapes

:*
dtype0
?
!Adadelta/dense_50/bias/accum_gradVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adadelta/dense_50/bias/accum_grad
?
5Adadelta/dense_50/bias/accum_grad/Read/ReadVariableOpReadVariableOp!Adadelta/dense_50/bias/accum_grad*
_output_shapes
:*
dtype0
?
#Adadelta/dense_51/kernel/accum_gradVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*4
shared_name%#Adadelta/dense_51/kernel/accum_grad
?
7Adadelta/dense_51/kernel/accum_grad/Read/ReadVariableOpReadVariableOp#Adadelta/dense_51/kernel/accum_grad*
_output_shapes

:*
dtype0
?
!Adadelta/dense_51/bias/accum_gradVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adadelta/dense_51/bias/accum_grad
?
5Adadelta/dense_51/bias/accum_grad/Read/ReadVariableOpReadVariableOp!Adadelta/dense_51/bias/accum_grad*
_output_shapes
:*
dtype0
?
0Adadelta/graph_conv_36/graph_conv_36_W/accum_varVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*A
shared_name20Adadelta/graph_conv_36/graph_conv_36_W/accum_var
?
DAdadelta/graph_conv_36/graph_conv_36_W/accum_var/Read/ReadVariableOpReadVariableOp0Adadelta/graph_conv_36/graph_conv_36_W/accum_var*
_output_shapes

:@*
dtype0
?
0Adadelta/graph_conv_36/graph_conv_36_b/accum_varVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*A
shared_name20Adadelta/graph_conv_36/graph_conv_36_b/accum_var
?
DAdadelta/graph_conv_36/graph_conv_36_b/accum_var/Read/ReadVariableOpReadVariableOp0Adadelta/graph_conv_36/graph_conv_36_b/accum_var*
_output_shapes
:@*
dtype0
?
0Adadelta/graph_conv_37/graph_conv_37_W/accum_varVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *A
shared_name20Adadelta/graph_conv_37/graph_conv_37_W/accum_var
?
DAdadelta/graph_conv_37/graph_conv_37_W/accum_var/Read/ReadVariableOpReadVariableOp0Adadelta/graph_conv_37/graph_conv_37_W/accum_var*
_output_shapes

:@ *
dtype0
?
0Adadelta/graph_conv_37/graph_conv_37_b/accum_varVarHandleOp*
_output_shapes
: *
dtype0*
shape: *A
shared_name20Adadelta/graph_conv_37/graph_conv_37_b/accum_var
?
DAdadelta/graph_conv_37/graph_conv_37_b/accum_var/Read/ReadVariableOpReadVariableOp0Adadelta/graph_conv_37/graph_conv_37_b/accum_var*
_output_shapes
: *
dtype0
?
0Adadelta/graph_conv_38/graph_conv_38_W/accum_varVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *A
shared_name20Adadelta/graph_conv_38/graph_conv_38_W/accum_var
?
DAdadelta/graph_conv_38/graph_conv_38_W/accum_var/Read/ReadVariableOpReadVariableOp0Adadelta/graph_conv_38/graph_conv_38_W/accum_var*
_output_shapes

: *
dtype0
?
0Adadelta/graph_conv_38/graph_conv_38_b/accum_varVarHandleOp*
_output_shapes
: *
dtype0*
shape:*A
shared_name20Adadelta/graph_conv_38/graph_conv_38_b/accum_var
?
DAdadelta/graph_conv_38/graph_conv_38_b/accum_var/Read/ReadVariableOpReadVariableOp0Adadelta/graph_conv_38/graph_conv_38_b/accum_var*
_output_shapes
:*
dtype0
?
*Adadelta/attention_12/att_weight/accum_varVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*;
shared_name,*Adadelta/attention_12/att_weight/accum_var
?
>Adadelta/attention_12/att_weight/accum_var/Read/ReadVariableOpReadVariableOp*Adadelta/attention_12/att_weight/accum_var*
_output_shapes

:*
dtype0
?
2Adadelta/neural_tensor_layer_12/Variable/accum_varVarHandleOp*
_output_shapes
: *
dtype0*
shape:*C
shared_name42Adadelta/neural_tensor_layer_12/Variable/accum_var
?
FAdadelta/neural_tensor_layer_12/Variable/accum_var/Read/ReadVariableOpReadVariableOp2Adadelta/neural_tensor_layer_12/Variable/accum_var*"
_output_shapes
:*
dtype0
?
4Adadelta/neural_tensor_layer_12/Variable/accum_var_1VarHandleOp*
_output_shapes
: *
dtype0*
shape
: *E
shared_name64Adadelta/neural_tensor_layer_12/Variable/accum_var_1
?
HAdadelta/neural_tensor_layer_12/Variable/accum_var_1/Read/ReadVariableOpReadVariableOp4Adadelta/neural_tensor_layer_12/Variable/accum_var_1*
_output_shapes

: *
dtype0
?
4Adadelta/neural_tensor_layer_12/Variable/accum_var_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:*E
shared_name64Adadelta/neural_tensor_layer_12/Variable/accum_var_2
?
HAdadelta/neural_tensor_layer_12/Variable/accum_var_2/Read/ReadVariableOpReadVariableOp4Adadelta/neural_tensor_layer_12/Variable/accum_var_2*
_output_shapes
:*
dtype0
?
"Adadelta/dense_48/kernel/accum_varVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*3
shared_name$"Adadelta/dense_48/kernel/accum_var
?
6Adadelta/dense_48/kernel/accum_var/Read/ReadVariableOpReadVariableOp"Adadelta/dense_48/kernel/accum_var*
_output_shapes

:*
dtype0
?
 Adadelta/dense_48/bias/accum_varVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adadelta/dense_48/bias/accum_var
?
4Adadelta/dense_48/bias/accum_var/Read/ReadVariableOpReadVariableOp Adadelta/dense_48/bias/accum_var*
_output_shapes
:*
dtype0
?
"Adadelta/dense_49/kernel/accum_varVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*3
shared_name$"Adadelta/dense_49/kernel/accum_var
?
6Adadelta/dense_49/kernel/accum_var/Read/ReadVariableOpReadVariableOp"Adadelta/dense_49/kernel/accum_var*
_output_shapes

:*
dtype0
?
 Adadelta/dense_49/bias/accum_varVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adadelta/dense_49/bias/accum_var
?
4Adadelta/dense_49/bias/accum_var/Read/ReadVariableOpReadVariableOp Adadelta/dense_49/bias/accum_var*
_output_shapes
:*
dtype0
?
"Adadelta/dense_50/kernel/accum_varVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*3
shared_name$"Adadelta/dense_50/kernel/accum_var
?
6Adadelta/dense_50/kernel/accum_var/Read/ReadVariableOpReadVariableOp"Adadelta/dense_50/kernel/accum_var*
_output_shapes

:*
dtype0
?
 Adadelta/dense_50/bias/accum_varVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adadelta/dense_50/bias/accum_var
?
4Adadelta/dense_50/bias/accum_var/Read/ReadVariableOpReadVariableOp Adadelta/dense_50/bias/accum_var*
_output_shapes
:*
dtype0
?
"Adadelta/dense_51/kernel/accum_varVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*3
shared_name$"Adadelta/dense_51/kernel/accum_var
?
6Adadelta/dense_51/kernel/accum_var/Read/ReadVariableOpReadVariableOp"Adadelta/dense_51/kernel/accum_var*
_output_shapes

:*
dtype0
?
 Adadelta/dense_51/bias/accum_varVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adadelta/dense_51/bias/accum_var
?
4Adadelta/dense_51/bias/accum_var/Read/ReadVariableOpReadVariableOp Adadelta/dense_51/bias/accum_var*
_output_shapes
:*
dtype0

NoOpNoOp
?j
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?i
value?iB?i B?i
?
layer-0
layer-1
layer-2
layer-3
layer_with_weights-0
layer-4
layer_with_weights-1
layer-5
layer_with_weights-2
layer-6
layer-7
	layer-8

layer_with_weights-3

layer-9
layer_with_weights-4
layer-10
layer_with_weights-5
layer-11
layer_with_weights-6
layer-12
layer_with_weights-7
layer-13
layer_with_weights-8
layer-14
layer-15
	optimizer
trainable_variables
	variables
regularization_losses
	keras_api

signatures
 
 
 
 
?
graph_conv_36_W
W
graph_conv_36_b
b
trainable_variables
	variables
regularization_losses
	keras_api
?
graph_conv_37_W
W
graph_conv_37_b
b
trainable_variables
 	variables
!regularization_losses
"	keras_api
?
#graph_conv_38_W
#W
$graph_conv_38_b
$b
%trainable_variables
&	variables
'regularization_losses
(	keras_api

)	keras_api

*	keras_api
s
+
att_weight
+weights_att
,trainable_variables
-	variables
.regularization_losses
/	keras_api

0W
1V
2b
3trainable_weights2
4trainable_variables
5	variables
6regularization_losses
7	keras_api
h

8kernel
9bias
:trainable_variables
;	variables
<regularization_losses
=	keras_api
h

>kernel
?bias
@trainable_variables
A	variables
Bregularization_losses
C	keras_api
h

Dkernel
Ebias
Ftrainable_variables
G	variables
Hregularization_losses
I	keras_api
h

Jkernel
Kbias
Ltrainable_variables
M	variables
Nregularization_losses
O	keras_api

P	keras_api
?
Qiter
	Rdecay
Slearning_rate
Trho
accum_grad?
accum_grad?
accum_grad?
accum_grad?#
accum_grad?$
accum_grad?+
accum_grad?0
accum_grad?1
accum_grad?2
accum_grad?8
accum_grad?9
accum_grad?>
accum_grad??
accum_grad?D
accum_grad?E
accum_grad?J
accum_grad?K
accum_grad?	accum_var?	accum_var?	accum_var?	accum_var?#	accum_var?$	accum_var?+	accum_var?0	accum_var?1	accum_var?2	accum_var?8	accum_var?9	accum_var?>	accum_var??	accum_var?D	accum_var?E	accum_var?J	accum_var?K	accum_var?
?
0
1
2
3
#4
$5
+6
07
18
29
810
911
>12
?13
D14
E15
J16
K17
?
0
1
2
3
#4
$5
+6
07
18
29
810
911
>12
?13
D14
E15
J16
K17
 
?
Unon_trainable_variables
trainable_variables
	variables
regularization_losses
Vlayer_metrics
Wlayer_regularization_losses

Xlayers
Ymetrics
 
rp
VARIABLE_VALUEgraph_conv_36/graph_conv_36_W?layer_with_weights-0/graph_conv_36_W/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEgraph_conv_36/graph_conv_36_b?layer_with_weights-0/graph_conv_36_b/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
Znon_trainable_variables
trainable_variables
	variables
regularization_losses
[layer_metrics
\layer_regularization_losses

]layers
^metrics
rp
VARIABLE_VALUEgraph_conv_37/graph_conv_37_W?layer_with_weights-1/graph_conv_37_W/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEgraph_conv_37/graph_conv_37_b?layer_with_weights-1/graph_conv_37_b/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
_non_trainable_variables
trainable_variables
 	variables
!regularization_losses
`layer_metrics
alayer_regularization_losses

blayers
cmetrics
rp
VARIABLE_VALUEgraph_conv_38/graph_conv_38_W?layer_with_weights-2/graph_conv_38_W/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEgraph_conv_38/graph_conv_38_b?layer_with_weights-2/graph_conv_38_b/.ATTRIBUTES/VARIABLE_VALUE

#0
$1

#0
$1
 
?
dnon_trainable_variables
%trainable_variables
&	variables
'regularization_losses
elayer_metrics
flayer_regularization_losses

glayers
hmetrics
 
 
ge
VARIABLE_VALUEattention_12/att_weight:layer_with_weights-3/att_weight/.ATTRIBUTES/VARIABLE_VALUE

+0

+0
 
?
inon_trainable_variables
,trainable_variables
-	variables
.regularization_losses
jlayer_metrics
klayer_regularization_losses

llayers
mmetrics
fd
VARIABLE_VALUEneural_tensor_layer_12/Variable1layer_with_weights-4/W/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUE!neural_tensor_layer_12/Variable_11layer_with_weights-4/V/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUE!neural_tensor_layer_12/Variable_21layer_with_weights-4/b/.ATTRIBUTES/VARIABLE_VALUE

00
11
22

00
11
22

00
11
22
 
?
nnon_trainable_variables
4trainable_variables
5	variables
6regularization_losses
olayer_metrics
player_regularization_losses

qlayers
rmetrics
[Y
VARIABLE_VALUEdense_48/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_48/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

80
91

80
91
 
?
snon_trainable_variables
:trainable_variables
;	variables
<regularization_losses
tlayer_metrics
ulayer_regularization_losses

vlayers
wmetrics
[Y
VARIABLE_VALUEdense_49/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_49/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

>0
?1

>0
?1
 
?
xnon_trainable_variables
@trainable_variables
A	variables
Bregularization_losses
ylayer_metrics
zlayer_regularization_losses

{layers
|metrics
[Y
VARIABLE_VALUEdense_50/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_50/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE

D0
E1

D0
E1
 
?
}non_trainable_variables
Ftrainable_variables
G	variables
Hregularization_losses
~layer_metrics
layer_regularization_losses
?layers
?metrics
[Y
VARIABLE_VALUEdense_51/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_51/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE

J0
K1

J0
K1
 
?
?non_trainable_variables
Ltrainable_variables
M	variables
Nregularization_losses
?layer_metrics
 ?layer_regularization_losses
?layers
?metrics
 
LJ
VARIABLE_VALUEAdadelta/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEAdadelta/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEAdadelta/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEAdadelta/rho(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUE
 
 
 
v
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15

?0
?1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

?total

?count
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
??
VARIABLE_VALUE1Adadelta/graph_conv_36/graph_conv_36_W/accum_graddlayer_with_weights-0/graph_conv_36_W/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE1Adadelta/graph_conv_36/graph_conv_36_b/accum_graddlayer_with_weights-0/graph_conv_36_b/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE1Adadelta/graph_conv_37/graph_conv_37_W/accum_graddlayer_with_weights-1/graph_conv_37_W/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE1Adadelta/graph_conv_37/graph_conv_37_b/accum_graddlayer_with_weights-1/graph_conv_37_b/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE1Adadelta/graph_conv_38/graph_conv_38_W/accum_graddlayer_with_weights-2/graph_conv_38_W/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE1Adadelta/graph_conv_38/graph_conv_38_b/accum_graddlayer_with_weights-2/graph_conv_38_b/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE+Adadelta/attention_12/att_weight/accum_grad_layer_with_weights-3/att_weight/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE3Adadelta/neural_tensor_layer_12/Variable/accum_gradVlayer_with_weights-4/W/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE5Adadelta/neural_tensor_layer_12/Variable/accum_grad_1Vlayer_with_weights-4/V/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE5Adadelta/neural_tensor_layer_12/Variable/accum_grad_2Vlayer_with_weights-4/b/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adadelta/dense_48/kernel/accum_grad[layer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!Adadelta/dense_48/bias/accum_gradYlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adadelta/dense_49/kernel/accum_grad[layer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!Adadelta/dense_49/bias/accum_gradYlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adadelta/dense_50/kernel/accum_grad[layer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!Adadelta/dense_50/bias/accum_gradYlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE#Adadelta/dense_51/kernel/accum_grad[layer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!Adadelta/dense_51/bias/accum_gradYlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE0Adadelta/graph_conv_36/graph_conv_36_W/accum_varclayer_with_weights-0/graph_conv_36_W/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE0Adadelta/graph_conv_36/graph_conv_36_b/accum_varclayer_with_weights-0/graph_conv_36_b/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE0Adadelta/graph_conv_37/graph_conv_37_W/accum_varclayer_with_weights-1/graph_conv_37_W/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE0Adadelta/graph_conv_37/graph_conv_37_b/accum_varclayer_with_weights-1/graph_conv_37_b/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE0Adadelta/graph_conv_38/graph_conv_38_W/accum_varclayer_with_weights-2/graph_conv_38_W/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE0Adadelta/graph_conv_38/graph_conv_38_b/accum_varclayer_with_weights-2/graph_conv_38_b/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adadelta/attention_12/att_weight/accum_var^layer_with_weights-3/att_weight/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE2Adadelta/neural_tensor_layer_12/Variable/accum_varUlayer_with_weights-4/W/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE4Adadelta/neural_tensor_layer_12/Variable/accum_var_1Ulayer_with_weights-4/V/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE4Adadelta/neural_tensor_layer_12/Variable/accum_var_2Ulayer_with_weights-4/b/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adadelta/dense_48/kernel/accum_varZlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE Adadelta/dense_48/bias/accum_varXlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adadelta/dense_49/kernel/accum_varZlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE Adadelta/dense_49/bias/accum_varXlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adadelta/dense_50/kernel/accum_varZlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE Adadelta/dense_50/bias/accum_varXlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adadelta/dense_51/kernel/accum_varZlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE Adadelta/dense_51/bias/accum_varXlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_input_49Placeholder*+
_output_shapes
:?????????	*
dtype0* 
shape:?????????	
?
serving_default_input_50Placeholder*=
_output_shapes+
):'???????????????????????????*
dtype0*2
shape):'???????????????????????????
?
serving_default_input_51Placeholder*+
_output_shapes
:?????????	*
dtype0* 
shape:?????????	
?
serving_default_input_52Placeholder*=
_output_shapes+
):'???????????????????????????*
dtype0*2
shape):'???????????????????????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_49serving_default_input_50serving_default_input_51serving_default_input_52graph_conv_36/graph_conv_36_Wgraph_conv_36/graph_conv_36_bgraph_conv_37/graph_conv_37_Wgraph_conv_37/graph_conv_37_bgraph_conv_38/graph_conv_38_Wgraph_conv_38/graph_conv_38_battention_12/att_weight!neural_tensor_layer_12/Variable_1neural_tensor_layer_12/Variable!neural_tensor_layer_12/Variable_2dense_48/kerneldense_48/biasdense_49/kerneldense_49/biasdense_50/kerneldense_50/biasdense_51/kerneldense_51/bias*!
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *.
f)R'
%__inference_signature_wrapper_4263724
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename1graph_conv_36/graph_conv_36_W/Read/ReadVariableOp1graph_conv_36/graph_conv_36_b/Read/ReadVariableOp1graph_conv_37/graph_conv_37_W/Read/ReadVariableOp1graph_conv_37/graph_conv_37_b/Read/ReadVariableOp1graph_conv_38/graph_conv_38_W/Read/ReadVariableOp1graph_conv_38/graph_conv_38_b/Read/ReadVariableOp+attention_12/att_weight/Read/ReadVariableOp3neural_tensor_layer_12/Variable/Read/ReadVariableOp5neural_tensor_layer_12/Variable_1/Read/ReadVariableOp5neural_tensor_layer_12/Variable_2/Read/ReadVariableOp#dense_48/kernel/Read/ReadVariableOp!dense_48/bias/Read/ReadVariableOp#dense_49/kernel/Read/ReadVariableOp!dense_49/bias/Read/ReadVariableOp#dense_50/kernel/Read/ReadVariableOp!dense_50/bias/Read/ReadVariableOp#dense_51/kernel/Read/ReadVariableOp!dense_51/bias/Read/ReadVariableOp!Adadelta/iter/Read/ReadVariableOp"Adadelta/decay/Read/ReadVariableOp*Adadelta/learning_rate/Read/ReadVariableOp Adadelta/rho/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOpEAdadelta/graph_conv_36/graph_conv_36_W/accum_grad/Read/ReadVariableOpEAdadelta/graph_conv_36/graph_conv_36_b/accum_grad/Read/ReadVariableOpEAdadelta/graph_conv_37/graph_conv_37_W/accum_grad/Read/ReadVariableOpEAdadelta/graph_conv_37/graph_conv_37_b/accum_grad/Read/ReadVariableOpEAdadelta/graph_conv_38/graph_conv_38_W/accum_grad/Read/ReadVariableOpEAdadelta/graph_conv_38/graph_conv_38_b/accum_grad/Read/ReadVariableOp?Adadelta/attention_12/att_weight/accum_grad/Read/ReadVariableOpGAdadelta/neural_tensor_layer_12/Variable/accum_grad/Read/ReadVariableOpIAdadelta/neural_tensor_layer_12/Variable/accum_grad_1/Read/ReadVariableOpIAdadelta/neural_tensor_layer_12/Variable/accum_grad_2/Read/ReadVariableOp7Adadelta/dense_48/kernel/accum_grad/Read/ReadVariableOp5Adadelta/dense_48/bias/accum_grad/Read/ReadVariableOp7Adadelta/dense_49/kernel/accum_grad/Read/ReadVariableOp5Adadelta/dense_49/bias/accum_grad/Read/ReadVariableOp7Adadelta/dense_50/kernel/accum_grad/Read/ReadVariableOp5Adadelta/dense_50/bias/accum_grad/Read/ReadVariableOp7Adadelta/dense_51/kernel/accum_grad/Read/ReadVariableOp5Adadelta/dense_51/bias/accum_grad/Read/ReadVariableOpDAdadelta/graph_conv_36/graph_conv_36_W/accum_var/Read/ReadVariableOpDAdadelta/graph_conv_36/graph_conv_36_b/accum_var/Read/ReadVariableOpDAdadelta/graph_conv_37/graph_conv_37_W/accum_var/Read/ReadVariableOpDAdadelta/graph_conv_37/graph_conv_37_b/accum_var/Read/ReadVariableOpDAdadelta/graph_conv_38/graph_conv_38_W/accum_var/Read/ReadVariableOpDAdadelta/graph_conv_38/graph_conv_38_b/accum_var/Read/ReadVariableOp>Adadelta/attention_12/att_weight/accum_var/Read/ReadVariableOpFAdadelta/neural_tensor_layer_12/Variable/accum_var/Read/ReadVariableOpHAdadelta/neural_tensor_layer_12/Variable/accum_var_1/Read/ReadVariableOpHAdadelta/neural_tensor_layer_12/Variable/accum_var_2/Read/ReadVariableOp6Adadelta/dense_48/kernel/accum_var/Read/ReadVariableOp4Adadelta/dense_48/bias/accum_var/Read/ReadVariableOp6Adadelta/dense_49/kernel/accum_var/Read/ReadVariableOp4Adadelta/dense_49/bias/accum_var/Read/ReadVariableOp6Adadelta/dense_50/kernel/accum_var/Read/ReadVariableOp4Adadelta/dense_50/bias/accum_var/Read/ReadVariableOp6Adadelta/dense_51/kernel/accum_var/Read/ReadVariableOp4Adadelta/dense_51/bias/accum_var/Read/ReadVariableOpConst*K
TinD
B2@	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference__traced_save_4265779
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamegraph_conv_36/graph_conv_36_Wgraph_conv_36/graph_conv_36_bgraph_conv_37/graph_conv_37_Wgraph_conv_37/graph_conv_37_bgraph_conv_38/graph_conv_38_Wgraph_conv_38/graph_conv_38_battention_12/att_weightneural_tensor_layer_12/Variable!neural_tensor_layer_12/Variable_1!neural_tensor_layer_12/Variable_2dense_48/kerneldense_48/biasdense_49/kerneldense_49/biasdense_50/kerneldense_50/biasdense_51/kerneldense_51/biasAdadelta/iterAdadelta/decayAdadelta/learning_rateAdadelta/rhototalcounttotal_1count_11Adadelta/graph_conv_36/graph_conv_36_W/accum_grad1Adadelta/graph_conv_36/graph_conv_36_b/accum_grad1Adadelta/graph_conv_37/graph_conv_37_W/accum_grad1Adadelta/graph_conv_37/graph_conv_37_b/accum_grad1Adadelta/graph_conv_38/graph_conv_38_W/accum_grad1Adadelta/graph_conv_38/graph_conv_38_b/accum_grad+Adadelta/attention_12/att_weight/accum_grad3Adadelta/neural_tensor_layer_12/Variable/accum_grad5Adadelta/neural_tensor_layer_12/Variable/accum_grad_15Adadelta/neural_tensor_layer_12/Variable/accum_grad_2#Adadelta/dense_48/kernel/accum_grad!Adadelta/dense_48/bias/accum_grad#Adadelta/dense_49/kernel/accum_grad!Adadelta/dense_49/bias/accum_grad#Adadelta/dense_50/kernel/accum_grad!Adadelta/dense_50/bias/accum_grad#Adadelta/dense_51/kernel/accum_grad!Adadelta/dense_51/bias/accum_grad0Adadelta/graph_conv_36/graph_conv_36_W/accum_var0Adadelta/graph_conv_36/graph_conv_36_b/accum_var0Adadelta/graph_conv_37/graph_conv_37_W/accum_var0Adadelta/graph_conv_37/graph_conv_37_b/accum_var0Adadelta/graph_conv_38/graph_conv_38_W/accum_var0Adadelta/graph_conv_38/graph_conv_38_b/accum_var*Adadelta/attention_12/att_weight/accum_var2Adadelta/neural_tensor_layer_12/Variable/accum_var4Adadelta/neural_tensor_layer_12/Variable/accum_var_14Adadelta/neural_tensor_layer_12/Variable/accum_var_2"Adadelta/dense_48/kernel/accum_var Adadelta/dense_48/bias/accum_var"Adadelta/dense_49/kernel/accum_var Adadelta/dense_49/bias/accum_var"Adadelta/dense_50/kernel/accum_var Adadelta/dense_50/bias/accum_var"Adadelta/dense_51/kernel/accum_var Adadelta/dense_51/bias/accum_var*J
TinC
A2?*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference__traced_restore_4265975??#
?

?
E__inference_dense_49_layer_call_and_return_conditional_losses_4262971

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOpj
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpx
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2	
BiasAddO
ReluReluBiasAdd:output:0*
T0*
_output_shapes

:2
Relud
IdentityIdentityRelu:activations:0^NoOp*
T0*
_output_shapes

:2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
:: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:F B

_output_shapes

:
 
_user_specified_nameinputs
?,
?
J__inference_graph_conv_38_layer_call_and_return_conditional_losses_4262691

inputs
inputs_11
shape_2_readvariableop_resource: +
add_1_readvariableop_resource:
identity??add_1/ReadVariableOp?transpose/ReadVariableOp}
MatMulBatchMatMulV2inputs_1inputs_1*
T0*=
_output_shapes+
):'???????????????????????????2
MatMulM
ShapeShapeMatMul:output:0*
T0*
_output_shapes
:2
Shapev
addAddV2MatMul:output:0inputs_1*
T0*=
_output_shapes+
):'???????????????????????????2
add[
	Greater/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
	Greater/y?
GreaterGreateradd:z:0Greater/y:output:0*
T0*=
_output_shapes+
):'???????????????????????????2	
Greaterx
CastCastGreater:z:0*

DstT0*

SrcT0
*=
_output_shapes+
):'???????????????????????????2
CastH
Shape_1Shapeinputs*
T0*
_output_shapes
:2	
Shape_1^
unstackUnpackShape_1:output:0*
T0*
_output_shapes
: : : *	
num2	
unstack?
Shape_2/ReadVariableOpReadVariableOpshape_2_readvariableop_resource*
_output_shapes

: *
dtype02
Shape_2/ReadVariableOpc
Shape_2Const*
_output_shapes
:*
dtype0*
valueB"       2	
Shape_2`
	unstack_1UnpackShape_2:output:0*
T0*
_output_shapes
: : *	
num2
	unstack_1o
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????    2
Reshape/shapeo
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:????????? 2	
Reshape?
transpose/ReadVariableOpReadVariableOpshape_2_readvariableop_resource*
_output_shapes

: *
dtype02
transpose/ReadVariableOpq
transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose/perm?
	transpose	Transpose transpose/ReadVariableOp:value:0transpose/perm:output:0*
T0*
_output_shapes

: 2
	transposes
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"    ????2
Reshape_1/shapes
	Reshape_1Reshapetranspose:y:0Reshape_1/shape:output:0*
T0*
_output_shapes

: 2
	Reshape_1v
MatMul_1MatMulReshape:output:0Reshape_1:output:0*
T0*'
_output_shapes
:?????????2

MatMul_1h
Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape_2/shape/2?
Reshape_2/shapePackunstack:output:0unstack:output:1Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape_2/shape?
	Reshape_2ReshapeMatMul_1:product:0Reshape_2/shape:output:0*
T0*4
_output_shapes"
 :??????????????????2
	Reshape_2?
add_1/ReadVariableOpReadVariableOpadd_1_readvariableop_resource*
_output_shapes
:*
dtype02
add_1/ReadVariableOp?
add_1AddV2Reshape_2:output:0add_1/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :??????????????????2
add_1?
MatMul_2BatchMatMulV2Cast:y:0Cast:y:0*
T0*=
_output_shapes+
):'???????????????????????????2

MatMul_2S
Shape_3ShapeMatMul_2:output:0*
T0*
_output_shapes
:2	
Shape_3|
add_2AddV2MatMul_2:output:0Cast:y:0*
T0*=
_output_shapes+
):'???????????????????????????2
add_2_
Greater_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
Greater_1/y?
	Greater_1Greater	add_2:z:0Greater_1/y:output:0*
T0*=
_output_shapes+
):'???????????????????????????2
	Greater_1~
Cast_1CastGreater_1:z:0*

DstT0*

SrcT0
*=
_output_shapes+
):'???????????????????????????2
Cast_1y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose
Cast_1:y:0transpose_1/perm:output:0*
T0*=
_output_shapes+
):'???????????????????????????2
transpose_1?
MatMul_3BatchMatMulV2transpose_1:y:0	add_1:z:0*
T0*4
_output_shapes"
 :??????????????????2

MatMul_3S
Shape_4ShapeMatMul_3:output:0*
T0*
_output_shapes
:2	
Shape_4p
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indices?
SumSum
Cast_1:y:0Sum/reduction_indices:output:0*
T0*4
_output_shapes"
 :??????????????????*
	keep_dims(2
SumW
add_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32	
add_3/yv
add_3AddV2Sum:output:0add_3/y:output:0*
T0*4
_output_shapes"
 :??????????????????2
add_3z
truedivRealDivMatMul_3:output:0	add_3:z:0*
T0*4
_output_shapes"
 :??????????????????2	
truediv`
ReluRelutruediv:z:0*
T0*4
_output_shapes"
 :??????????????????2
Reluz
IdentityIdentityRelu:activations:0^NoOp*
T0*4
_output_shapes"
 :??????????????????2

Identity?
NoOpNoOp^add_1/ReadVariableOp^transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*`
_input_shapesO
M:?????????????????? :'???????????????????????????: : 2,
add_1/ReadVariableOpadd_1/ReadVariableOp24
transpose/ReadVariableOptranspose/ReadVariableOp:\ X
4
_output_shapes"
 :?????????????????? 
 
_user_specified_nameinputs:ea
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?,
?
J__inference_graph_conv_37_layer_call_and_return_conditional_losses_4263247

inputs
inputs_11
shape_2_readvariableop_resource:@ +
add_1_readvariableop_resource: 
identity??add_1/ReadVariableOp?transpose/ReadVariableOp}
MatMulBatchMatMulV2inputs_1inputs_1*
T0*=
_output_shapes+
):'???????????????????????????2
MatMulM
ShapeShapeMatMul:output:0*
T0*
_output_shapes
:2
Shapev
addAddV2MatMul:output:0inputs_1*
T0*=
_output_shapes+
):'???????????????????????????2
add[
	Greater/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
	Greater/y?
GreaterGreateradd:z:0Greater/y:output:0*
T0*=
_output_shapes+
):'???????????????????????????2	
Greaterx
CastCastGreater:z:0*

DstT0*

SrcT0
*=
_output_shapes+
):'???????????????????????????2
CastH
Shape_1Shapeinputs*
T0*
_output_shapes
:2	
Shape_1^
unstackUnpackShape_1:output:0*
T0*
_output_shapes
: : : *	
num2	
unstack?
Shape_2/ReadVariableOpReadVariableOpshape_2_readvariableop_resource*
_output_shapes

:@ *
dtype02
Shape_2/ReadVariableOpc
Shape_2Const*
_output_shapes
:*
dtype0*
valueB"@       2	
Shape_2`
	unstack_1UnpackShape_2:output:0*
T0*
_output_shapes
: : *	
num2
	unstack_1o
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   2
Reshape/shapeo
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:?????????@2	
Reshape?
transpose/ReadVariableOpReadVariableOpshape_2_readvariableop_resource*
_output_shapes

:@ *
dtype02
transpose/ReadVariableOpq
transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose/perm?
	transpose	Transpose transpose/ReadVariableOp:value:0transpose/perm:output:0*
T0*
_output_shapes

:@ 2
	transposes
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"@   ????2
Reshape_1/shapes
	Reshape_1Reshapetranspose:y:0Reshape_1/shape:output:0*
T0*
_output_shapes

:@ 2
	Reshape_1v
MatMul_1MatMulReshape:output:0Reshape_1:output:0*
T0*'
_output_shapes
:????????? 2

MatMul_1h
Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B : 2
Reshape_2/shape/2?
Reshape_2/shapePackunstack:output:0unstack:output:1Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape_2/shape?
	Reshape_2ReshapeMatMul_1:product:0Reshape_2/shape:output:0*
T0*4
_output_shapes"
 :?????????????????? 2
	Reshape_2?
add_1/ReadVariableOpReadVariableOpadd_1_readvariableop_resource*
_output_shapes
: *
dtype02
add_1/ReadVariableOp?
add_1AddV2Reshape_2:output:0add_1/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :?????????????????? 2
add_1?
MatMul_2BatchMatMulV2Cast:y:0Cast:y:0*
T0*=
_output_shapes+
):'???????????????????????????2

MatMul_2S
Shape_3ShapeMatMul_2:output:0*
T0*
_output_shapes
:2	
Shape_3|
add_2AddV2MatMul_2:output:0Cast:y:0*
T0*=
_output_shapes+
):'???????????????????????????2
add_2_
Greater_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
Greater_1/y?
	Greater_1Greater	add_2:z:0Greater_1/y:output:0*
T0*=
_output_shapes+
):'???????????????????????????2
	Greater_1~
Cast_1CastGreater_1:z:0*

DstT0*

SrcT0
*=
_output_shapes+
):'???????????????????????????2
Cast_1y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose
Cast_1:y:0transpose_1/perm:output:0*
T0*=
_output_shapes+
):'???????????????????????????2
transpose_1?
MatMul_3BatchMatMulV2transpose_1:y:0	add_1:z:0*
T0*4
_output_shapes"
 :?????????????????? 2

MatMul_3S
Shape_4ShapeMatMul_3:output:0*
T0*
_output_shapes
:2	
Shape_4p
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indices?
SumSum
Cast_1:y:0Sum/reduction_indices:output:0*
T0*4
_output_shapes"
 :??????????????????*
	keep_dims(2
SumW
add_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32	
add_3/yv
add_3AddV2Sum:output:0add_3/y:output:0*
T0*4
_output_shapes"
 :??????????????????2
add_3z
truedivRealDivMatMul_3:output:0	add_3:z:0*
T0*4
_output_shapes"
 :?????????????????? 2	
truediv`
ReluRelutruediv:z:0*
T0*4
_output_shapes"
 :?????????????????? 2
Reluz
IdentityIdentityRelu:activations:0^NoOp*
T0*4
_output_shapes"
 :?????????????????? 2

Identity?
NoOpNoOp^add_1/ReadVariableOp^transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*`
_input_shapesO
M:??????????????????@:'???????????????????????????: : 2,
add_1/ReadVariableOpadd_1/ReadVariableOp24
transpose/ReadVariableOptranspose/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????@
 
_user_specified_nameinputs:ea
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
??
?
E__inference_model_12_layer_call_and_return_conditional_losses_4264886
inputs_0
inputs_1
inputs_2
inputs_3?
-graph_conv_36_shape_2_readvariableop_resource:@9
+graph_conv_36_add_1_readvariableop_resource:@?
-graph_conv_37_shape_2_readvariableop_resource:@ 9
+graph_conv_37_add_1_readvariableop_resource: ?
-graph_conv_38_shape_2_readvariableop_resource: 9
+graph_conv_38_add_1_readvariableop_resource:=
+attention_12_matmul_readvariableop_resource:G
5neural_tensor_layer_12_matmul_readvariableop_resource: D
.neural_tensor_layer_12_readvariableop_resource:@
2neural_tensor_layer_12_add_readvariableop_resource:9
'dense_48_matmul_readvariableop_resource:6
(dense_48_biasadd_readvariableop_resource:9
'dense_49_matmul_readvariableop_resource:6
(dense_49_biasadd_readvariableop_resource:9
'dense_50_matmul_readvariableop_resource:6
(dense_50_biasadd_readvariableop_resource:9
'dense_51_matmul_readvariableop_resource:6
(dense_51_biasadd_readvariableop_resource:
identity??"attention_12/MatMul/ReadVariableOp?$attention_12/MatMul_3/ReadVariableOp?dense_48/BiasAdd/ReadVariableOp?dense_48/MatMul/ReadVariableOp?dense_49/BiasAdd/ReadVariableOp?dense_49/MatMul/ReadVariableOp?dense_50/BiasAdd/ReadVariableOp?dense_50/MatMul/ReadVariableOp?dense_51/BiasAdd/ReadVariableOp?dense_51/MatMul/ReadVariableOp?"graph_conv_36/add_1/ReadVariableOp?"graph_conv_36/add_5/ReadVariableOp?&graph_conv_36/transpose/ReadVariableOp?(graph_conv_36/transpose_2/ReadVariableOp?"graph_conv_37/add_1/ReadVariableOp?"graph_conv_37/add_5/ReadVariableOp?&graph_conv_37/transpose/ReadVariableOp?(graph_conv_37/transpose_2/ReadVariableOp?"graph_conv_38/add_1/ReadVariableOp?"graph_conv_38/add_5/ReadVariableOp?&graph_conv_38/transpose/ReadVariableOp?(graph_conv_38/transpose_2/ReadVariableOp?,neural_tensor_layer_12/MatMul/ReadVariableOp?%neural_tensor_layer_12/ReadVariableOp?'neural_tensor_layer_12/ReadVariableOp_1?(neural_tensor_layer_12/ReadVariableOp_10?(neural_tensor_layer_12/ReadVariableOp_11?(neural_tensor_layer_12/ReadVariableOp_12?(neural_tensor_layer_12/ReadVariableOp_13?(neural_tensor_layer_12/ReadVariableOp_14?(neural_tensor_layer_12/ReadVariableOp_15?'neural_tensor_layer_12/ReadVariableOp_2?'neural_tensor_layer_12/ReadVariableOp_3?'neural_tensor_layer_12/ReadVariableOp_4?'neural_tensor_layer_12/ReadVariableOp_5?'neural_tensor_layer_12/ReadVariableOp_6?'neural_tensor_layer_12/ReadVariableOp_7?'neural_tensor_layer_12/ReadVariableOp_8?'neural_tensor_layer_12/ReadVariableOp_9?)neural_tensor_layer_12/add/ReadVariableOp?+neural_tensor_layer_12/add_1/ReadVariableOp?,neural_tensor_layer_12/add_10/ReadVariableOp?,neural_tensor_layer_12/add_11/ReadVariableOp?,neural_tensor_layer_12/add_12/ReadVariableOp?,neural_tensor_layer_12/add_13/ReadVariableOp?,neural_tensor_layer_12/add_14/ReadVariableOp?,neural_tensor_layer_12/add_15/ReadVariableOp?+neural_tensor_layer_12/add_2/ReadVariableOp?+neural_tensor_layer_12/add_3/ReadVariableOp?+neural_tensor_layer_12/add_4/ReadVariableOp?+neural_tensor_layer_12/add_5/ReadVariableOp?+neural_tensor_layer_12/add_6/ReadVariableOp?+neural_tensor_layer_12/add_7/ReadVariableOp?+neural_tensor_layer_12/add_8/ReadVariableOp?+neural_tensor_layer_12/add_9/ReadVariableOp?
graph_conv_36/MatMulBatchMatMulV2inputs_3inputs_3*
T0*=
_output_shapes+
):'???????????????????????????2
graph_conv_36/MatMulw
graph_conv_36/ShapeShapegraph_conv_36/MatMul:output:0*
T0*
_output_shapes
:2
graph_conv_36/Shape?
graph_conv_36/addAddV2graph_conv_36/MatMul:output:0inputs_3*
T0*=
_output_shapes+
):'???????????????????????????2
graph_conv_36/addw
graph_conv_36/Greater/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
graph_conv_36/Greater/y?
graph_conv_36/GreaterGreatergraph_conv_36/add:z:0 graph_conv_36/Greater/y:output:0*
T0*=
_output_shapes+
):'???????????????????????????2
graph_conv_36/Greater?
graph_conv_36/CastCastgraph_conv_36/Greater:z:0*

DstT0*

SrcT0
*=
_output_shapes+
):'???????????????????????????2
graph_conv_36/Castf
graph_conv_36/Shape_1Shapeinputs_2*
T0*
_output_shapes
:2
graph_conv_36/Shape_1?
graph_conv_36/unstackUnpackgraph_conv_36/Shape_1:output:0*
T0*
_output_shapes
: : : *	
num2
graph_conv_36/unstack?
$graph_conv_36/Shape_2/ReadVariableOpReadVariableOp-graph_conv_36_shape_2_readvariableop_resource*
_output_shapes

:@*
dtype02&
$graph_conv_36/Shape_2/ReadVariableOp
graph_conv_36/Shape_2Const*
_output_shapes
:*
dtype0*
valueB"   @   2
graph_conv_36/Shape_2?
graph_conv_36/unstack_1Unpackgraph_conv_36/Shape_2:output:0*
T0*
_output_shapes
: : *	
num2
graph_conv_36/unstack_1?
graph_conv_36/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
graph_conv_36/Reshape/shape?
graph_conv_36/ReshapeReshapeinputs_2$graph_conv_36/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
graph_conv_36/Reshape?
&graph_conv_36/transpose/ReadVariableOpReadVariableOp-graph_conv_36_shape_2_readvariableop_resource*
_output_shapes

:@*
dtype02(
&graph_conv_36/transpose/ReadVariableOp?
graph_conv_36/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2
graph_conv_36/transpose/perm?
graph_conv_36/transpose	Transpose.graph_conv_36/transpose/ReadVariableOp:value:0%graph_conv_36/transpose/perm:output:0*
T0*
_output_shapes

:@2
graph_conv_36/transpose?
graph_conv_36/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ????2
graph_conv_36/Reshape_1/shape?
graph_conv_36/Reshape_1Reshapegraph_conv_36/transpose:y:0&graph_conv_36/Reshape_1/shape:output:0*
T0*
_output_shapes

:@2
graph_conv_36/Reshape_1?
graph_conv_36/MatMul_1MatMulgraph_conv_36/Reshape:output:0 graph_conv_36/Reshape_1:output:0*
T0*'
_output_shapes
:?????????@2
graph_conv_36/MatMul_1?
graph_conv_36/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :	2!
graph_conv_36/Reshape_2/shape/1?
graph_conv_36/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :@2!
graph_conv_36/Reshape_2/shape/2?
graph_conv_36/Reshape_2/shapePackgraph_conv_36/unstack:output:0(graph_conv_36/Reshape_2/shape/1:output:0(graph_conv_36/Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:2
graph_conv_36/Reshape_2/shape?
graph_conv_36/Reshape_2Reshape graph_conv_36/MatMul_1:product:0&graph_conv_36/Reshape_2/shape:output:0*
T0*+
_output_shapes
:?????????	@2
graph_conv_36/Reshape_2?
"graph_conv_36/add_1/ReadVariableOpReadVariableOp+graph_conv_36_add_1_readvariableop_resource*
_output_shapes
:@*
dtype02$
"graph_conv_36/add_1/ReadVariableOp?
graph_conv_36/add_1AddV2 graph_conv_36/Reshape_2:output:0*graph_conv_36/add_1/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????	@2
graph_conv_36/add_1?
graph_conv_36/MatMul_2BatchMatMulV2graph_conv_36/Cast:y:0graph_conv_36/Cast:y:0*
T0*=
_output_shapes+
):'???????????????????????????2
graph_conv_36/MatMul_2}
graph_conv_36/Shape_3Shapegraph_conv_36/MatMul_2:output:0*
T0*
_output_shapes
:2
graph_conv_36/Shape_3?
graph_conv_36/add_2AddV2graph_conv_36/MatMul_2:output:0graph_conv_36/Cast:y:0*
T0*=
_output_shapes+
):'???????????????????????????2
graph_conv_36/add_2{
graph_conv_36/Greater_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
graph_conv_36/Greater_1/y?
graph_conv_36/Greater_1Greatergraph_conv_36/add_2:z:0"graph_conv_36/Greater_1/y:output:0*
T0*=
_output_shapes+
):'???????????????????????????2
graph_conv_36/Greater_1?
graph_conv_36/Cast_1Castgraph_conv_36/Greater_1:z:0*

DstT0*

SrcT0
*=
_output_shapes+
):'???????????????????????????2
graph_conv_36/Cast_1?
graph_conv_36/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2 
graph_conv_36/transpose_1/perm?
graph_conv_36/transpose_1	Transposegraph_conv_36/Cast_1:y:0'graph_conv_36/transpose_1/perm:output:0*
T0*=
_output_shapes+
):'???????????????????????????2
graph_conv_36/transpose_1?
graph_conv_36/MatMul_3BatchMatMulV2graph_conv_36/transpose_1:y:0graph_conv_36/add_1:z:0*
T0*4
_output_shapes"
 :??????????????????@2
graph_conv_36/MatMul_3}
graph_conv_36/Shape_4Shapegraph_conv_36/MatMul_3:output:0*
T0*
_output_shapes
:2
graph_conv_36/Shape_4?
#graph_conv_36/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2%
#graph_conv_36/Sum/reduction_indices?
graph_conv_36/SumSumgraph_conv_36/Cast_1:y:0,graph_conv_36/Sum/reduction_indices:output:0*
T0*4
_output_shapes"
 :??????????????????*
	keep_dims(2
graph_conv_36/Sums
graph_conv_36/add_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
graph_conv_36/add_3/y?
graph_conv_36/add_3AddV2graph_conv_36/Sum:output:0graph_conv_36/add_3/y:output:0*
T0*4
_output_shapes"
 :??????????????????2
graph_conv_36/add_3?
graph_conv_36/truedivRealDivgraph_conv_36/MatMul_3:output:0graph_conv_36/add_3:z:0*
T0*4
_output_shapes"
 :??????????????????@2
graph_conv_36/truediv?
graph_conv_36/ReluRelugraph_conv_36/truediv:z:0*
T0*4
_output_shapes"
 :??????????????????@2
graph_conv_36/Relu?
graph_conv_36/MatMul_4BatchMatMulV2inputs_1inputs_1*
T0*=
_output_shapes+
):'???????????????????????????2
graph_conv_36/MatMul_4}
graph_conv_36/Shape_5Shapegraph_conv_36/MatMul_4:output:0*
T0*
_output_shapes
:2
graph_conv_36/Shape_5?
graph_conv_36/add_4AddV2graph_conv_36/MatMul_4:output:0inputs_1*
T0*=
_output_shapes+
):'???????????????????????????2
graph_conv_36/add_4{
graph_conv_36/Greater_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
graph_conv_36/Greater_2/y?
graph_conv_36/Greater_2Greatergraph_conv_36/add_4:z:0"graph_conv_36/Greater_2/y:output:0*
T0*=
_output_shapes+
):'???????????????????????????2
graph_conv_36/Greater_2?
graph_conv_36/Cast_2Castgraph_conv_36/Greater_2:z:0*

DstT0*

SrcT0
*=
_output_shapes+
):'???????????????????????????2
graph_conv_36/Cast_2f
graph_conv_36/Shape_6Shapeinputs_0*
T0*
_output_shapes
:2
graph_conv_36/Shape_6?
graph_conv_36/unstack_2Unpackgraph_conv_36/Shape_6:output:0*
T0*
_output_shapes
: : : *	
num2
graph_conv_36/unstack_2?
$graph_conv_36/Shape_7/ReadVariableOpReadVariableOp-graph_conv_36_shape_2_readvariableop_resource*
_output_shapes

:@*
dtype02&
$graph_conv_36/Shape_7/ReadVariableOp
graph_conv_36/Shape_7Const*
_output_shapes
:*
dtype0*
valueB"   @   2
graph_conv_36/Shape_7?
graph_conv_36/unstack_3Unpackgraph_conv_36/Shape_7:output:0*
T0*
_output_shapes
: : *	
num2
graph_conv_36/unstack_3?
graph_conv_36/Reshape_3/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
graph_conv_36/Reshape_3/shape?
graph_conv_36/Reshape_3Reshapeinputs_0&graph_conv_36/Reshape_3/shape:output:0*
T0*'
_output_shapes
:?????????2
graph_conv_36/Reshape_3?
(graph_conv_36/transpose_2/ReadVariableOpReadVariableOp-graph_conv_36_shape_2_readvariableop_resource*
_output_shapes

:@*
dtype02*
(graph_conv_36/transpose_2/ReadVariableOp?
graph_conv_36/transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2 
graph_conv_36/transpose_2/perm?
graph_conv_36/transpose_2	Transpose0graph_conv_36/transpose_2/ReadVariableOp:value:0'graph_conv_36/transpose_2/perm:output:0*
T0*
_output_shapes

:@2
graph_conv_36/transpose_2?
graph_conv_36/Reshape_4/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ????2
graph_conv_36/Reshape_4/shape?
graph_conv_36/Reshape_4Reshapegraph_conv_36/transpose_2:y:0&graph_conv_36/Reshape_4/shape:output:0*
T0*
_output_shapes

:@2
graph_conv_36/Reshape_4?
graph_conv_36/MatMul_5MatMul graph_conv_36/Reshape_3:output:0 graph_conv_36/Reshape_4:output:0*
T0*'
_output_shapes
:?????????@2
graph_conv_36/MatMul_5?
graph_conv_36/Reshape_5/shape/1Const*
_output_shapes
: *
dtype0*
value	B :	2!
graph_conv_36/Reshape_5/shape/1?
graph_conv_36/Reshape_5/shape/2Const*
_output_shapes
: *
dtype0*
value	B :@2!
graph_conv_36/Reshape_5/shape/2?
graph_conv_36/Reshape_5/shapePack graph_conv_36/unstack_2:output:0(graph_conv_36/Reshape_5/shape/1:output:0(graph_conv_36/Reshape_5/shape/2:output:0*
N*
T0*
_output_shapes
:2
graph_conv_36/Reshape_5/shape?
graph_conv_36/Reshape_5Reshape graph_conv_36/MatMul_5:product:0&graph_conv_36/Reshape_5/shape:output:0*
T0*+
_output_shapes
:?????????	@2
graph_conv_36/Reshape_5?
"graph_conv_36/add_5/ReadVariableOpReadVariableOp+graph_conv_36_add_1_readvariableop_resource*
_output_shapes
:@*
dtype02$
"graph_conv_36/add_5/ReadVariableOp?
graph_conv_36/add_5AddV2 graph_conv_36/Reshape_5:output:0*graph_conv_36/add_5/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????	@2
graph_conv_36/add_5?
graph_conv_36/MatMul_6BatchMatMulV2graph_conv_36/Cast_2:y:0graph_conv_36/Cast_2:y:0*
T0*=
_output_shapes+
):'???????????????????????????2
graph_conv_36/MatMul_6}
graph_conv_36/Shape_8Shapegraph_conv_36/MatMul_6:output:0*
T0*
_output_shapes
:2
graph_conv_36/Shape_8?
graph_conv_36/add_6AddV2graph_conv_36/MatMul_6:output:0graph_conv_36/Cast_2:y:0*
T0*=
_output_shapes+
):'???????????????????????????2
graph_conv_36/add_6{
graph_conv_36/Greater_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
graph_conv_36/Greater_3/y?
graph_conv_36/Greater_3Greatergraph_conv_36/add_6:z:0"graph_conv_36/Greater_3/y:output:0*
T0*=
_output_shapes+
):'???????????????????????????2
graph_conv_36/Greater_3?
graph_conv_36/Cast_3Castgraph_conv_36/Greater_3:z:0*

DstT0*

SrcT0
*=
_output_shapes+
):'???????????????????????????2
graph_conv_36/Cast_3?
graph_conv_36/transpose_3/permConst*
_output_shapes
:*
dtype0*!
valueB"          2 
graph_conv_36/transpose_3/perm?
graph_conv_36/transpose_3	Transposegraph_conv_36/Cast_3:y:0'graph_conv_36/transpose_3/perm:output:0*
T0*=
_output_shapes+
):'???????????????????????????2
graph_conv_36/transpose_3?
graph_conv_36/MatMul_7BatchMatMulV2graph_conv_36/transpose_3:y:0graph_conv_36/add_5:z:0*
T0*4
_output_shapes"
 :??????????????????@2
graph_conv_36/MatMul_7}
graph_conv_36/Shape_9Shapegraph_conv_36/MatMul_7:output:0*
T0*
_output_shapes
:2
graph_conv_36/Shape_9?
%graph_conv_36/Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2'
%graph_conv_36/Sum_1/reduction_indices?
graph_conv_36/Sum_1Sumgraph_conv_36/Cast_3:y:0.graph_conv_36/Sum_1/reduction_indices:output:0*
T0*4
_output_shapes"
 :??????????????????*
	keep_dims(2
graph_conv_36/Sum_1s
graph_conv_36/add_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
graph_conv_36/add_7/y?
graph_conv_36/add_7AddV2graph_conv_36/Sum_1:output:0graph_conv_36/add_7/y:output:0*
T0*4
_output_shapes"
 :??????????????????2
graph_conv_36/add_7?
graph_conv_36/truediv_1RealDivgraph_conv_36/MatMul_7:output:0graph_conv_36/add_7:z:0*
T0*4
_output_shapes"
 :??????????????????@2
graph_conv_36/truediv_1?
graph_conv_36/Relu_1Relugraph_conv_36/truediv_1:z:0*
T0*4
_output_shapes"
 :??????????????????@2
graph_conv_36/Relu_1?
graph_conv_37/MatMulBatchMatMulV2inputs_3inputs_3*
T0*=
_output_shapes+
):'???????????????????????????2
graph_conv_37/MatMulw
graph_conv_37/ShapeShapegraph_conv_37/MatMul:output:0*
T0*
_output_shapes
:2
graph_conv_37/Shape?
graph_conv_37/addAddV2graph_conv_37/MatMul:output:0inputs_3*
T0*=
_output_shapes+
):'???????????????????????????2
graph_conv_37/addw
graph_conv_37/Greater/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
graph_conv_37/Greater/y?
graph_conv_37/GreaterGreatergraph_conv_37/add:z:0 graph_conv_37/Greater/y:output:0*
T0*=
_output_shapes+
):'???????????????????????????2
graph_conv_37/Greater?
graph_conv_37/CastCastgraph_conv_37/Greater:z:0*

DstT0*

SrcT0
*=
_output_shapes+
):'???????????????????????????2
graph_conv_37/Cast~
graph_conv_37/Shape_1Shape graph_conv_36/Relu:activations:0*
T0*
_output_shapes
:2
graph_conv_37/Shape_1?
graph_conv_37/unstackUnpackgraph_conv_37/Shape_1:output:0*
T0*
_output_shapes
: : : *	
num2
graph_conv_37/unstack?
$graph_conv_37/Shape_2/ReadVariableOpReadVariableOp-graph_conv_37_shape_2_readvariableop_resource*
_output_shapes

:@ *
dtype02&
$graph_conv_37/Shape_2/ReadVariableOp
graph_conv_37/Shape_2Const*
_output_shapes
:*
dtype0*
valueB"@       2
graph_conv_37/Shape_2?
graph_conv_37/unstack_1Unpackgraph_conv_37/Shape_2:output:0*
T0*
_output_shapes
: : *	
num2
graph_conv_37/unstack_1?
graph_conv_37/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   2
graph_conv_37/Reshape/shape?
graph_conv_37/ReshapeReshape graph_conv_36/Relu:activations:0$graph_conv_37/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????@2
graph_conv_37/Reshape?
&graph_conv_37/transpose/ReadVariableOpReadVariableOp-graph_conv_37_shape_2_readvariableop_resource*
_output_shapes

:@ *
dtype02(
&graph_conv_37/transpose/ReadVariableOp?
graph_conv_37/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2
graph_conv_37/transpose/perm?
graph_conv_37/transpose	Transpose.graph_conv_37/transpose/ReadVariableOp:value:0%graph_conv_37/transpose/perm:output:0*
T0*
_output_shapes

:@ 2
graph_conv_37/transpose?
graph_conv_37/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"@   ????2
graph_conv_37/Reshape_1/shape?
graph_conv_37/Reshape_1Reshapegraph_conv_37/transpose:y:0&graph_conv_37/Reshape_1/shape:output:0*
T0*
_output_shapes

:@ 2
graph_conv_37/Reshape_1?
graph_conv_37/MatMul_1MatMulgraph_conv_37/Reshape:output:0 graph_conv_37/Reshape_1:output:0*
T0*'
_output_shapes
:????????? 2
graph_conv_37/MatMul_1?
graph_conv_37/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B : 2!
graph_conv_37/Reshape_2/shape/2?
graph_conv_37/Reshape_2/shapePackgraph_conv_37/unstack:output:0graph_conv_37/unstack:output:1(graph_conv_37/Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:2
graph_conv_37/Reshape_2/shape?
graph_conv_37/Reshape_2Reshape graph_conv_37/MatMul_1:product:0&graph_conv_37/Reshape_2/shape:output:0*
T0*4
_output_shapes"
 :?????????????????? 2
graph_conv_37/Reshape_2?
"graph_conv_37/add_1/ReadVariableOpReadVariableOp+graph_conv_37_add_1_readvariableop_resource*
_output_shapes
: *
dtype02$
"graph_conv_37/add_1/ReadVariableOp?
graph_conv_37/add_1AddV2 graph_conv_37/Reshape_2:output:0*graph_conv_37/add_1/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :?????????????????? 2
graph_conv_37/add_1?
graph_conv_37/MatMul_2BatchMatMulV2graph_conv_37/Cast:y:0graph_conv_37/Cast:y:0*
T0*=
_output_shapes+
):'???????????????????????????2
graph_conv_37/MatMul_2}
graph_conv_37/Shape_3Shapegraph_conv_37/MatMul_2:output:0*
T0*
_output_shapes
:2
graph_conv_37/Shape_3?
graph_conv_37/add_2AddV2graph_conv_37/MatMul_2:output:0graph_conv_37/Cast:y:0*
T0*=
_output_shapes+
):'???????????????????????????2
graph_conv_37/add_2{
graph_conv_37/Greater_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
graph_conv_37/Greater_1/y?
graph_conv_37/Greater_1Greatergraph_conv_37/add_2:z:0"graph_conv_37/Greater_1/y:output:0*
T0*=
_output_shapes+
):'???????????????????????????2
graph_conv_37/Greater_1?
graph_conv_37/Cast_1Castgraph_conv_37/Greater_1:z:0*

DstT0*

SrcT0
*=
_output_shapes+
):'???????????????????????????2
graph_conv_37/Cast_1?
graph_conv_37/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2 
graph_conv_37/transpose_1/perm?
graph_conv_37/transpose_1	Transposegraph_conv_37/Cast_1:y:0'graph_conv_37/transpose_1/perm:output:0*
T0*=
_output_shapes+
):'???????????????????????????2
graph_conv_37/transpose_1?
graph_conv_37/MatMul_3BatchMatMulV2graph_conv_37/transpose_1:y:0graph_conv_37/add_1:z:0*
T0*4
_output_shapes"
 :?????????????????? 2
graph_conv_37/MatMul_3}
graph_conv_37/Shape_4Shapegraph_conv_37/MatMul_3:output:0*
T0*
_output_shapes
:2
graph_conv_37/Shape_4?
#graph_conv_37/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2%
#graph_conv_37/Sum/reduction_indices?
graph_conv_37/SumSumgraph_conv_37/Cast_1:y:0,graph_conv_37/Sum/reduction_indices:output:0*
T0*4
_output_shapes"
 :??????????????????*
	keep_dims(2
graph_conv_37/Sums
graph_conv_37/add_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
graph_conv_37/add_3/y?
graph_conv_37/add_3AddV2graph_conv_37/Sum:output:0graph_conv_37/add_3/y:output:0*
T0*4
_output_shapes"
 :??????????????????2
graph_conv_37/add_3?
graph_conv_37/truedivRealDivgraph_conv_37/MatMul_3:output:0graph_conv_37/add_3:z:0*
T0*4
_output_shapes"
 :?????????????????? 2
graph_conv_37/truediv?
graph_conv_37/ReluRelugraph_conv_37/truediv:z:0*
T0*4
_output_shapes"
 :?????????????????? 2
graph_conv_37/Relu?
graph_conv_37/MatMul_4BatchMatMulV2inputs_1inputs_1*
T0*=
_output_shapes+
):'???????????????????????????2
graph_conv_37/MatMul_4}
graph_conv_37/Shape_5Shapegraph_conv_37/MatMul_4:output:0*
T0*
_output_shapes
:2
graph_conv_37/Shape_5?
graph_conv_37/add_4AddV2graph_conv_37/MatMul_4:output:0inputs_1*
T0*=
_output_shapes+
):'???????????????????????????2
graph_conv_37/add_4{
graph_conv_37/Greater_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
graph_conv_37/Greater_2/y?
graph_conv_37/Greater_2Greatergraph_conv_37/add_4:z:0"graph_conv_37/Greater_2/y:output:0*
T0*=
_output_shapes+
):'???????????????????????????2
graph_conv_37/Greater_2?
graph_conv_37/Cast_2Castgraph_conv_37/Greater_2:z:0*

DstT0*

SrcT0
*=
_output_shapes+
):'???????????????????????????2
graph_conv_37/Cast_2?
graph_conv_37/Shape_6Shape"graph_conv_36/Relu_1:activations:0*
T0*
_output_shapes
:2
graph_conv_37/Shape_6?
graph_conv_37/unstack_2Unpackgraph_conv_37/Shape_6:output:0*
T0*
_output_shapes
: : : *	
num2
graph_conv_37/unstack_2?
$graph_conv_37/Shape_7/ReadVariableOpReadVariableOp-graph_conv_37_shape_2_readvariableop_resource*
_output_shapes

:@ *
dtype02&
$graph_conv_37/Shape_7/ReadVariableOp
graph_conv_37/Shape_7Const*
_output_shapes
:*
dtype0*
valueB"@       2
graph_conv_37/Shape_7?
graph_conv_37/unstack_3Unpackgraph_conv_37/Shape_7:output:0*
T0*
_output_shapes
: : *	
num2
graph_conv_37/unstack_3?
graph_conv_37/Reshape_3/shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   2
graph_conv_37/Reshape_3/shape?
graph_conv_37/Reshape_3Reshape"graph_conv_36/Relu_1:activations:0&graph_conv_37/Reshape_3/shape:output:0*
T0*'
_output_shapes
:?????????@2
graph_conv_37/Reshape_3?
(graph_conv_37/transpose_2/ReadVariableOpReadVariableOp-graph_conv_37_shape_2_readvariableop_resource*
_output_shapes

:@ *
dtype02*
(graph_conv_37/transpose_2/ReadVariableOp?
graph_conv_37/transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2 
graph_conv_37/transpose_2/perm?
graph_conv_37/transpose_2	Transpose0graph_conv_37/transpose_2/ReadVariableOp:value:0'graph_conv_37/transpose_2/perm:output:0*
T0*
_output_shapes

:@ 2
graph_conv_37/transpose_2?
graph_conv_37/Reshape_4/shapeConst*
_output_shapes
:*
dtype0*
valueB"@   ????2
graph_conv_37/Reshape_4/shape?
graph_conv_37/Reshape_4Reshapegraph_conv_37/transpose_2:y:0&graph_conv_37/Reshape_4/shape:output:0*
T0*
_output_shapes

:@ 2
graph_conv_37/Reshape_4?
graph_conv_37/MatMul_5MatMul graph_conv_37/Reshape_3:output:0 graph_conv_37/Reshape_4:output:0*
T0*'
_output_shapes
:????????? 2
graph_conv_37/MatMul_5?
graph_conv_37/Reshape_5/shape/2Const*
_output_shapes
: *
dtype0*
value	B : 2!
graph_conv_37/Reshape_5/shape/2?
graph_conv_37/Reshape_5/shapePack graph_conv_37/unstack_2:output:0 graph_conv_37/unstack_2:output:1(graph_conv_37/Reshape_5/shape/2:output:0*
N*
T0*
_output_shapes
:2
graph_conv_37/Reshape_5/shape?
graph_conv_37/Reshape_5Reshape graph_conv_37/MatMul_5:product:0&graph_conv_37/Reshape_5/shape:output:0*
T0*4
_output_shapes"
 :?????????????????? 2
graph_conv_37/Reshape_5?
"graph_conv_37/add_5/ReadVariableOpReadVariableOp+graph_conv_37_add_1_readvariableop_resource*
_output_shapes
: *
dtype02$
"graph_conv_37/add_5/ReadVariableOp?
graph_conv_37/add_5AddV2 graph_conv_37/Reshape_5:output:0*graph_conv_37/add_5/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :?????????????????? 2
graph_conv_37/add_5?
graph_conv_37/MatMul_6BatchMatMulV2graph_conv_37/Cast_2:y:0graph_conv_37/Cast_2:y:0*
T0*=
_output_shapes+
):'???????????????????????????2
graph_conv_37/MatMul_6}
graph_conv_37/Shape_8Shapegraph_conv_37/MatMul_6:output:0*
T0*
_output_shapes
:2
graph_conv_37/Shape_8?
graph_conv_37/add_6AddV2graph_conv_37/MatMul_6:output:0graph_conv_37/Cast_2:y:0*
T0*=
_output_shapes+
):'???????????????????????????2
graph_conv_37/add_6{
graph_conv_37/Greater_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
graph_conv_37/Greater_3/y?
graph_conv_37/Greater_3Greatergraph_conv_37/add_6:z:0"graph_conv_37/Greater_3/y:output:0*
T0*=
_output_shapes+
):'???????????????????????????2
graph_conv_37/Greater_3?
graph_conv_37/Cast_3Castgraph_conv_37/Greater_3:z:0*

DstT0*

SrcT0
*=
_output_shapes+
):'???????????????????????????2
graph_conv_37/Cast_3?
graph_conv_37/transpose_3/permConst*
_output_shapes
:*
dtype0*!
valueB"          2 
graph_conv_37/transpose_3/perm?
graph_conv_37/transpose_3	Transposegraph_conv_37/Cast_3:y:0'graph_conv_37/transpose_3/perm:output:0*
T0*=
_output_shapes+
):'???????????????????????????2
graph_conv_37/transpose_3?
graph_conv_37/MatMul_7BatchMatMulV2graph_conv_37/transpose_3:y:0graph_conv_37/add_5:z:0*
T0*4
_output_shapes"
 :?????????????????? 2
graph_conv_37/MatMul_7}
graph_conv_37/Shape_9Shapegraph_conv_37/MatMul_7:output:0*
T0*
_output_shapes
:2
graph_conv_37/Shape_9?
%graph_conv_37/Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2'
%graph_conv_37/Sum_1/reduction_indices?
graph_conv_37/Sum_1Sumgraph_conv_37/Cast_3:y:0.graph_conv_37/Sum_1/reduction_indices:output:0*
T0*4
_output_shapes"
 :??????????????????*
	keep_dims(2
graph_conv_37/Sum_1s
graph_conv_37/add_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
graph_conv_37/add_7/y?
graph_conv_37/add_7AddV2graph_conv_37/Sum_1:output:0graph_conv_37/add_7/y:output:0*
T0*4
_output_shapes"
 :??????????????????2
graph_conv_37/add_7?
graph_conv_37/truediv_1RealDivgraph_conv_37/MatMul_7:output:0graph_conv_37/add_7:z:0*
T0*4
_output_shapes"
 :?????????????????? 2
graph_conv_37/truediv_1?
graph_conv_37/Relu_1Relugraph_conv_37/truediv_1:z:0*
T0*4
_output_shapes"
 :?????????????????? 2
graph_conv_37/Relu_1?
graph_conv_38/MatMulBatchMatMulV2inputs_3inputs_3*
T0*=
_output_shapes+
):'???????????????????????????2
graph_conv_38/MatMulw
graph_conv_38/ShapeShapegraph_conv_38/MatMul:output:0*
T0*
_output_shapes
:2
graph_conv_38/Shape?
graph_conv_38/addAddV2graph_conv_38/MatMul:output:0inputs_3*
T0*=
_output_shapes+
):'???????????????????????????2
graph_conv_38/addw
graph_conv_38/Greater/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
graph_conv_38/Greater/y?
graph_conv_38/GreaterGreatergraph_conv_38/add:z:0 graph_conv_38/Greater/y:output:0*
T0*=
_output_shapes+
):'???????????????????????????2
graph_conv_38/Greater?
graph_conv_38/CastCastgraph_conv_38/Greater:z:0*

DstT0*

SrcT0
*=
_output_shapes+
):'???????????????????????????2
graph_conv_38/Cast~
graph_conv_38/Shape_1Shape graph_conv_37/Relu:activations:0*
T0*
_output_shapes
:2
graph_conv_38/Shape_1?
graph_conv_38/unstackUnpackgraph_conv_38/Shape_1:output:0*
T0*
_output_shapes
: : : *	
num2
graph_conv_38/unstack?
$graph_conv_38/Shape_2/ReadVariableOpReadVariableOp-graph_conv_38_shape_2_readvariableop_resource*
_output_shapes

: *
dtype02&
$graph_conv_38/Shape_2/ReadVariableOp
graph_conv_38/Shape_2Const*
_output_shapes
:*
dtype0*
valueB"       2
graph_conv_38/Shape_2?
graph_conv_38/unstack_1Unpackgraph_conv_38/Shape_2:output:0*
T0*
_output_shapes
: : *	
num2
graph_conv_38/unstack_1?
graph_conv_38/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????    2
graph_conv_38/Reshape/shape?
graph_conv_38/ReshapeReshape graph_conv_37/Relu:activations:0$graph_conv_38/Reshape/shape:output:0*
T0*'
_output_shapes
:????????? 2
graph_conv_38/Reshape?
&graph_conv_38/transpose/ReadVariableOpReadVariableOp-graph_conv_38_shape_2_readvariableop_resource*
_output_shapes

: *
dtype02(
&graph_conv_38/transpose/ReadVariableOp?
graph_conv_38/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2
graph_conv_38/transpose/perm?
graph_conv_38/transpose	Transpose.graph_conv_38/transpose/ReadVariableOp:value:0%graph_conv_38/transpose/perm:output:0*
T0*
_output_shapes

: 2
graph_conv_38/transpose?
graph_conv_38/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"    ????2
graph_conv_38/Reshape_1/shape?
graph_conv_38/Reshape_1Reshapegraph_conv_38/transpose:y:0&graph_conv_38/Reshape_1/shape:output:0*
T0*
_output_shapes

: 2
graph_conv_38/Reshape_1?
graph_conv_38/MatMul_1MatMulgraph_conv_38/Reshape:output:0 graph_conv_38/Reshape_1:output:0*
T0*'
_output_shapes
:?????????2
graph_conv_38/MatMul_1?
graph_conv_38/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2!
graph_conv_38/Reshape_2/shape/2?
graph_conv_38/Reshape_2/shapePackgraph_conv_38/unstack:output:0graph_conv_38/unstack:output:1(graph_conv_38/Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:2
graph_conv_38/Reshape_2/shape?
graph_conv_38/Reshape_2Reshape graph_conv_38/MatMul_1:product:0&graph_conv_38/Reshape_2/shape:output:0*
T0*4
_output_shapes"
 :??????????????????2
graph_conv_38/Reshape_2?
"graph_conv_38/add_1/ReadVariableOpReadVariableOp+graph_conv_38_add_1_readvariableop_resource*
_output_shapes
:*
dtype02$
"graph_conv_38/add_1/ReadVariableOp?
graph_conv_38/add_1AddV2 graph_conv_38/Reshape_2:output:0*graph_conv_38/add_1/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :??????????????????2
graph_conv_38/add_1?
graph_conv_38/MatMul_2BatchMatMulV2graph_conv_38/Cast:y:0graph_conv_38/Cast:y:0*
T0*=
_output_shapes+
):'???????????????????????????2
graph_conv_38/MatMul_2}
graph_conv_38/Shape_3Shapegraph_conv_38/MatMul_2:output:0*
T0*
_output_shapes
:2
graph_conv_38/Shape_3?
graph_conv_38/add_2AddV2graph_conv_38/MatMul_2:output:0graph_conv_38/Cast:y:0*
T0*=
_output_shapes+
):'???????????????????????????2
graph_conv_38/add_2{
graph_conv_38/Greater_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
graph_conv_38/Greater_1/y?
graph_conv_38/Greater_1Greatergraph_conv_38/add_2:z:0"graph_conv_38/Greater_1/y:output:0*
T0*=
_output_shapes+
):'???????????????????????????2
graph_conv_38/Greater_1?
graph_conv_38/Cast_1Castgraph_conv_38/Greater_1:z:0*

DstT0*

SrcT0
*=
_output_shapes+
):'???????????????????????????2
graph_conv_38/Cast_1?
graph_conv_38/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2 
graph_conv_38/transpose_1/perm?
graph_conv_38/transpose_1	Transposegraph_conv_38/Cast_1:y:0'graph_conv_38/transpose_1/perm:output:0*
T0*=
_output_shapes+
):'???????????????????????????2
graph_conv_38/transpose_1?
graph_conv_38/MatMul_3BatchMatMulV2graph_conv_38/transpose_1:y:0graph_conv_38/add_1:z:0*
T0*4
_output_shapes"
 :??????????????????2
graph_conv_38/MatMul_3}
graph_conv_38/Shape_4Shapegraph_conv_38/MatMul_3:output:0*
T0*
_output_shapes
:2
graph_conv_38/Shape_4?
#graph_conv_38/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2%
#graph_conv_38/Sum/reduction_indices?
graph_conv_38/SumSumgraph_conv_38/Cast_1:y:0,graph_conv_38/Sum/reduction_indices:output:0*
T0*4
_output_shapes"
 :??????????????????*
	keep_dims(2
graph_conv_38/Sums
graph_conv_38/add_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
graph_conv_38/add_3/y?
graph_conv_38/add_3AddV2graph_conv_38/Sum:output:0graph_conv_38/add_3/y:output:0*
T0*4
_output_shapes"
 :??????????????????2
graph_conv_38/add_3?
graph_conv_38/truedivRealDivgraph_conv_38/MatMul_3:output:0graph_conv_38/add_3:z:0*
T0*4
_output_shapes"
 :??????????????????2
graph_conv_38/truediv?
graph_conv_38/ReluRelugraph_conv_38/truediv:z:0*
T0*4
_output_shapes"
 :??????????????????2
graph_conv_38/Relu?
graph_conv_38/MatMul_4BatchMatMulV2inputs_1inputs_1*
T0*=
_output_shapes+
):'???????????????????????????2
graph_conv_38/MatMul_4}
graph_conv_38/Shape_5Shapegraph_conv_38/MatMul_4:output:0*
T0*
_output_shapes
:2
graph_conv_38/Shape_5?
graph_conv_38/add_4AddV2graph_conv_38/MatMul_4:output:0inputs_1*
T0*=
_output_shapes+
):'???????????????????????????2
graph_conv_38/add_4{
graph_conv_38/Greater_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
graph_conv_38/Greater_2/y?
graph_conv_38/Greater_2Greatergraph_conv_38/add_4:z:0"graph_conv_38/Greater_2/y:output:0*
T0*=
_output_shapes+
):'???????????????????????????2
graph_conv_38/Greater_2?
graph_conv_38/Cast_2Castgraph_conv_38/Greater_2:z:0*

DstT0*

SrcT0
*=
_output_shapes+
):'???????????????????????????2
graph_conv_38/Cast_2?
graph_conv_38/Shape_6Shape"graph_conv_37/Relu_1:activations:0*
T0*
_output_shapes
:2
graph_conv_38/Shape_6?
graph_conv_38/unstack_2Unpackgraph_conv_38/Shape_6:output:0*
T0*
_output_shapes
: : : *	
num2
graph_conv_38/unstack_2?
$graph_conv_38/Shape_7/ReadVariableOpReadVariableOp-graph_conv_38_shape_2_readvariableop_resource*
_output_shapes

: *
dtype02&
$graph_conv_38/Shape_7/ReadVariableOp
graph_conv_38/Shape_7Const*
_output_shapes
:*
dtype0*
valueB"       2
graph_conv_38/Shape_7?
graph_conv_38/unstack_3Unpackgraph_conv_38/Shape_7:output:0*
T0*
_output_shapes
: : *	
num2
graph_conv_38/unstack_3?
graph_conv_38/Reshape_3/shapeConst*
_output_shapes
:*
dtype0*
valueB"????    2
graph_conv_38/Reshape_3/shape?
graph_conv_38/Reshape_3Reshape"graph_conv_37/Relu_1:activations:0&graph_conv_38/Reshape_3/shape:output:0*
T0*'
_output_shapes
:????????? 2
graph_conv_38/Reshape_3?
(graph_conv_38/transpose_2/ReadVariableOpReadVariableOp-graph_conv_38_shape_2_readvariableop_resource*
_output_shapes

: *
dtype02*
(graph_conv_38/transpose_2/ReadVariableOp?
graph_conv_38/transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2 
graph_conv_38/transpose_2/perm?
graph_conv_38/transpose_2	Transpose0graph_conv_38/transpose_2/ReadVariableOp:value:0'graph_conv_38/transpose_2/perm:output:0*
T0*
_output_shapes

: 2
graph_conv_38/transpose_2?
graph_conv_38/Reshape_4/shapeConst*
_output_shapes
:*
dtype0*
valueB"    ????2
graph_conv_38/Reshape_4/shape?
graph_conv_38/Reshape_4Reshapegraph_conv_38/transpose_2:y:0&graph_conv_38/Reshape_4/shape:output:0*
T0*
_output_shapes

: 2
graph_conv_38/Reshape_4?
graph_conv_38/MatMul_5MatMul graph_conv_38/Reshape_3:output:0 graph_conv_38/Reshape_4:output:0*
T0*'
_output_shapes
:?????????2
graph_conv_38/MatMul_5?
graph_conv_38/Reshape_5/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2!
graph_conv_38/Reshape_5/shape/2?
graph_conv_38/Reshape_5/shapePack graph_conv_38/unstack_2:output:0 graph_conv_38/unstack_2:output:1(graph_conv_38/Reshape_5/shape/2:output:0*
N*
T0*
_output_shapes
:2
graph_conv_38/Reshape_5/shape?
graph_conv_38/Reshape_5Reshape graph_conv_38/MatMul_5:product:0&graph_conv_38/Reshape_5/shape:output:0*
T0*4
_output_shapes"
 :??????????????????2
graph_conv_38/Reshape_5?
"graph_conv_38/add_5/ReadVariableOpReadVariableOp+graph_conv_38_add_1_readvariableop_resource*
_output_shapes
:*
dtype02$
"graph_conv_38/add_5/ReadVariableOp?
graph_conv_38/add_5AddV2 graph_conv_38/Reshape_5:output:0*graph_conv_38/add_5/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :??????????????????2
graph_conv_38/add_5?
graph_conv_38/MatMul_6BatchMatMulV2graph_conv_38/Cast_2:y:0graph_conv_38/Cast_2:y:0*
T0*=
_output_shapes+
):'???????????????????????????2
graph_conv_38/MatMul_6}
graph_conv_38/Shape_8Shapegraph_conv_38/MatMul_6:output:0*
T0*
_output_shapes
:2
graph_conv_38/Shape_8?
graph_conv_38/add_6AddV2graph_conv_38/MatMul_6:output:0graph_conv_38/Cast_2:y:0*
T0*=
_output_shapes+
):'???????????????????????????2
graph_conv_38/add_6{
graph_conv_38/Greater_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
graph_conv_38/Greater_3/y?
graph_conv_38/Greater_3Greatergraph_conv_38/add_6:z:0"graph_conv_38/Greater_3/y:output:0*
T0*=
_output_shapes+
):'???????????????????????????2
graph_conv_38/Greater_3?
graph_conv_38/Cast_3Castgraph_conv_38/Greater_3:z:0*

DstT0*

SrcT0
*=
_output_shapes+
):'???????????????????????????2
graph_conv_38/Cast_3?
graph_conv_38/transpose_3/permConst*
_output_shapes
:*
dtype0*!
valueB"          2 
graph_conv_38/transpose_3/perm?
graph_conv_38/transpose_3	Transposegraph_conv_38/Cast_3:y:0'graph_conv_38/transpose_3/perm:output:0*
T0*=
_output_shapes+
):'???????????????????????????2
graph_conv_38/transpose_3?
graph_conv_38/MatMul_7BatchMatMulV2graph_conv_38/transpose_3:y:0graph_conv_38/add_5:z:0*
T0*4
_output_shapes"
 :??????????????????2
graph_conv_38/MatMul_7}
graph_conv_38/Shape_9Shapegraph_conv_38/MatMul_7:output:0*
T0*
_output_shapes
:2
graph_conv_38/Shape_9?
%graph_conv_38/Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2'
%graph_conv_38/Sum_1/reduction_indices?
graph_conv_38/Sum_1Sumgraph_conv_38/Cast_3:y:0.graph_conv_38/Sum_1/reduction_indices:output:0*
T0*4
_output_shapes"
 :??????????????????*
	keep_dims(2
graph_conv_38/Sum_1s
graph_conv_38/add_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
graph_conv_38/add_7/y?
graph_conv_38/add_7AddV2graph_conv_38/Sum_1:output:0graph_conv_38/add_7/y:output:0*
T0*4
_output_shapes"
 :??????????????????2
graph_conv_38/add_7?
graph_conv_38/truediv_1RealDivgraph_conv_38/MatMul_7:output:0graph_conv_38/add_7:z:0*
T0*4
_output_shapes"
 :??????????????????2
graph_conv_38/truediv_1?
graph_conv_38/Relu_1Relugraph_conv_38/truediv_1:z:0*
T0*4
_output_shapes"
 :??????????????????2
graph_conv_38/Relu_1?
/tf.__operators__.getitem_25/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/tf.__operators__.getitem_25/strided_slice/stack?
1tf.__operators__.getitem_25/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1tf.__operators__.getitem_25/strided_slice/stack_1?
1tf.__operators__.getitem_25/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1tf.__operators__.getitem_25/strided_slice/stack_2?
)tf.__operators__.getitem_25/strided_sliceStridedSlice graph_conv_38/Relu:activations:08tf.__operators__.getitem_25/strided_slice/stack:output:0:tf.__operators__.getitem_25/strided_slice/stack_1:output:0:tf.__operators__.getitem_25/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2+
)tf.__operators__.getitem_25/strided_slice?
/tf.__operators__.getitem_24/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/tf.__operators__.getitem_24/strided_slice/stack?
1tf.__operators__.getitem_24/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1tf.__operators__.getitem_24/strided_slice/stack_1?
1tf.__operators__.getitem_24/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1tf.__operators__.getitem_24/strided_slice/stack_2?
)tf.__operators__.getitem_24/strided_sliceStridedSlice"graph_conv_38/Relu_1:activations:08tf.__operators__.getitem_24/strided_slice/stack:output:0:tf.__operators__.getitem_24/strided_slice/stack_1:output:0:tf.__operators__.getitem_24/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2+
)tf.__operators__.getitem_24/strided_slice?
"attention_12/MatMul/ReadVariableOpReadVariableOp+attention_12_matmul_readvariableop_resource*
_output_shapes

:*
dtype02$
"attention_12/MatMul/ReadVariableOp?
attention_12/MatMulMatMul2tf.__operators__.getitem_24/strided_slice:output:0*attention_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
attention_12/MatMul?
#attention_12/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2%
#attention_12/Mean/reduction_indices?
attention_12/MeanMeanattention_12/MatMul:product:0,attention_12/Mean/reduction_indices:output:0*
T0*
_output_shapes
:2
attention_12/Meano
attention_12/TanhTanhattention_12/Mean:output:0*
T0*
_output_shapes
:2
attention_12/Tanh?
attention_12/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ????2
attention_12/Reshape/shape?
attention_12/ReshapeReshapeattention_12/Tanh:y:0#attention_12/Reshape/shape:output:0*
T0*
_output_shapes

:2
attention_12/Reshape?
attention_12/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2
attention_12/transpose/perm?
attention_12/transpose	Transposeattention_12/Reshape:output:0$attention_12/transpose/perm:output:0*
T0*
_output_shapes

:2
attention_12/transpose?
attention_12/MatMul_1MatMul2tf.__operators__.getitem_24/strided_slice:output:0attention_12/transpose:y:0*
T0*'
_output_shapes
:?????????2
attention_12/MatMul_1?
attention_12/SigmoidSigmoidattention_12/MatMul_1:product:0*
T0*'
_output_shapes
:?????????2
attention_12/Sigmoid?
attention_12/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
attention_12/transpose_1/perm?
attention_12/transpose_1	Transpose2tf.__operators__.getitem_24/strided_slice:output:0&attention_12/transpose_1/perm:output:0*
T0*'
_output_shapes
:?????????2
attention_12/transpose_1?
attention_12/MatMul_2MatMulattention_12/transpose_1:y:0attention_12/Sigmoid:y:0*
T0*
_output_shapes

:2
attention_12/MatMul_2?
attention_12/transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2
attention_12/transpose_2/perm?
attention_12/transpose_2	Transposeattention_12/MatMul_2:product:0&attention_12/transpose_2/perm:output:0*
T0*
_output_shapes

:2
attention_12/transpose_2?
$attention_12/MatMul_3/ReadVariableOpReadVariableOp+attention_12_matmul_readvariableop_resource*
_output_shapes

:*
dtype02&
$attention_12/MatMul_3/ReadVariableOp?
attention_12/MatMul_3MatMul2tf.__operators__.getitem_25/strided_slice:output:0,attention_12/MatMul_3/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
attention_12/MatMul_3?
%attention_12/Mean_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2'
%attention_12/Mean_1/reduction_indices?
attention_12/Mean_1Meanattention_12/MatMul_3:product:0.attention_12/Mean_1/reduction_indices:output:0*
T0*
_output_shapes
:2
attention_12/Mean_1u
attention_12/Tanh_1Tanhattention_12/Mean_1:output:0*
T0*
_output_shapes
:2
attention_12/Tanh_1?
attention_12/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ????2
attention_12/Reshape_1/shape?
attention_12/Reshape_1Reshapeattention_12/Tanh_1:y:0%attention_12/Reshape_1/shape:output:0*
T0*
_output_shapes

:2
attention_12/Reshape_1?
attention_12/transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
attention_12/transpose_3/perm?
attention_12/transpose_3	Transposeattention_12/Reshape_1:output:0&attention_12/transpose_3/perm:output:0*
T0*
_output_shapes

:2
attention_12/transpose_3?
attention_12/MatMul_4MatMul2tf.__operators__.getitem_25/strided_slice:output:0attention_12/transpose_3:y:0*
T0*'
_output_shapes
:?????????2
attention_12/MatMul_4?
attention_12/Sigmoid_1Sigmoidattention_12/MatMul_4:product:0*
T0*'
_output_shapes
:?????????2
attention_12/Sigmoid_1?
attention_12/transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       2
attention_12/transpose_4/perm?
attention_12/transpose_4	Transpose2tf.__operators__.getitem_25/strided_slice:output:0&attention_12/transpose_4/perm:output:0*
T0*'
_output_shapes
:?????????2
attention_12/transpose_4?
attention_12/MatMul_5MatMulattention_12/transpose_4:y:0attention_12/Sigmoid_1:y:0*
T0*
_output_shapes

:2
attention_12/MatMul_5?
attention_12/transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       2
attention_12/transpose_5/perm?
attention_12/transpose_5	Transposeattention_12/MatMul_5:product:0&attention_12/transpose_5/perm:output:0*
T0*
_output_shapes

:2
attention_12/transpose_5?
neural_tensor_layer_12/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2
neural_tensor_layer_12/Shape?
*neural_tensor_layer_12/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*neural_tensor_layer_12/strided_slice/stack?
,neural_tensor_layer_12/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,neural_tensor_layer_12/strided_slice/stack_1?
,neural_tensor_layer_12/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,neural_tensor_layer_12/strided_slice/stack_2?
$neural_tensor_layer_12/strided_sliceStridedSlice%neural_tensor_layer_12/Shape:output:03neural_tensor_layer_12/strided_slice/stack:output:05neural_tensor_layer_12/strided_slice/stack_1:output:05neural_tensor_layer_12/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$neural_tensor_layer_12/strided_slice?
"neural_tensor_layer_12/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2$
"neural_tensor_layer_12/concat/axis?
neural_tensor_layer_12/concatConcatV2attention_12/transpose_2:y:0attention_12/transpose_5:y:0+neural_tensor_layer_12/concat/axis:output:0*
N*
T0*
_output_shapes

: 2
neural_tensor_layer_12/concat?
,neural_tensor_layer_12/MatMul/ReadVariableOpReadVariableOp5neural_tensor_layer_12_matmul_readvariableop_resource*
_output_shapes

: *
dtype02.
,neural_tensor_layer_12/MatMul/ReadVariableOp?
neural_tensor_layer_12/MatMulMatMul&neural_tensor_layer_12/concat:output:04neural_tensor_layer_12/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
neural_tensor_layer_12/MatMul?
%neural_tensor_layer_12/ReadVariableOpReadVariableOp.neural_tensor_layer_12_readvariableop_resource*"
_output_shapes
:*
dtype02'
%neural_tensor_layer_12/ReadVariableOp?
,neural_tensor_layer_12/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,neural_tensor_layer_12/strided_slice_1/stack?
.neural_tensor_layer_12/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.neural_tensor_layer_12/strided_slice_1/stack_1?
.neural_tensor_layer_12/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.neural_tensor_layer_12/strided_slice_1/stack_2?
&neural_tensor_layer_12/strided_slice_1StridedSlice-neural_tensor_layer_12/ReadVariableOp:value:05neural_tensor_layer_12/strided_slice_1/stack:output:07neural_tensor_layer_12/strided_slice_1/stack_1:output:07neural_tensor_layer_12/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2(
&neural_tensor_layer_12/strided_slice_1?
neural_tensor_layer_12/MatMul_1MatMulattention_12/transpose_2:y:0/neural_tensor_layer_12/strided_slice_1:output:0*
T0*
_output_shapes

:2!
neural_tensor_layer_12/MatMul_1?
neural_tensor_layer_12/mulMulattention_12/transpose_5:y:0)neural_tensor_layer_12/MatMul_1:product:0*
T0*
_output_shapes

:2
neural_tensor_layer_12/mul?
)neural_tensor_layer_12/add/ReadVariableOpReadVariableOp2neural_tensor_layer_12_add_readvariableop_resource*
_output_shapes
:*
dtype02+
)neural_tensor_layer_12/add/ReadVariableOp?
neural_tensor_layer_12/addAddV2neural_tensor_layer_12/mul:z:01neural_tensor_layer_12/add/ReadVariableOp:value:0*
T0*
_output_shapes

:2
neural_tensor_layer_12/add?
,neural_tensor_layer_12/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2.
,neural_tensor_layer_12/Sum/reduction_indices?
neural_tensor_layer_12/SumSumneural_tensor_layer_12/add:z:05neural_tensor_layer_12/Sum/reduction_indices:output:0*
T0*
_output_shapes
:2
neural_tensor_layer_12/Sum?
'neural_tensor_layer_12/ReadVariableOp_1ReadVariableOp.neural_tensor_layer_12_readvariableop_resource*"
_output_shapes
:*
dtype02)
'neural_tensor_layer_12/ReadVariableOp_1?
,neural_tensor_layer_12/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2.
,neural_tensor_layer_12/strided_slice_2/stack?
.neural_tensor_layer_12/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.neural_tensor_layer_12/strided_slice_2/stack_1?
.neural_tensor_layer_12/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.neural_tensor_layer_12/strided_slice_2/stack_2?
&neural_tensor_layer_12/strided_slice_2StridedSlice/neural_tensor_layer_12/ReadVariableOp_1:value:05neural_tensor_layer_12/strided_slice_2/stack:output:07neural_tensor_layer_12/strided_slice_2/stack_1:output:07neural_tensor_layer_12/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2(
&neural_tensor_layer_12/strided_slice_2?
neural_tensor_layer_12/MatMul_2MatMulattention_12/transpose_2:y:0/neural_tensor_layer_12/strided_slice_2:output:0*
T0*
_output_shapes

:2!
neural_tensor_layer_12/MatMul_2?
neural_tensor_layer_12/mul_1Mulattention_12/transpose_5:y:0)neural_tensor_layer_12/MatMul_2:product:0*
T0*
_output_shapes

:2
neural_tensor_layer_12/mul_1?
+neural_tensor_layer_12/add_1/ReadVariableOpReadVariableOp2neural_tensor_layer_12_add_readvariableop_resource*
_output_shapes
:*
dtype02-
+neural_tensor_layer_12/add_1/ReadVariableOp?
neural_tensor_layer_12/add_1AddV2 neural_tensor_layer_12/mul_1:z:03neural_tensor_layer_12/add_1/ReadVariableOp:value:0*
T0*
_output_shapes

:2
neural_tensor_layer_12/add_1?
.neural_tensor_layer_12/Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :20
.neural_tensor_layer_12/Sum_1/reduction_indices?
neural_tensor_layer_12/Sum_1Sum neural_tensor_layer_12/add_1:z:07neural_tensor_layer_12/Sum_1/reduction_indices:output:0*
T0*
_output_shapes
:2
neural_tensor_layer_12/Sum_1?
'neural_tensor_layer_12/ReadVariableOp_2ReadVariableOp.neural_tensor_layer_12_readvariableop_resource*"
_output_shapes
:*
dtype02)
'neural_tensor_layer_12/ReadVariableOp_2?
,neural_tensor_layer_12/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:2.
,neural_tensor_layer_12/strided_slice_3/stack?
.neural_tensor_layer_12/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.neural_tensor_layer_12/strided_slice_3/stack_1?
.neural_tensor_layer_12/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.neural_tensor_layer_12/strided_slice_3/stack_2?
&neural_tensor_layer_12/strided_slice_3StridedSlice/neural_tensor_layer_12/ReadVariableOp_2:value:05neural_tensor_layer_12/strided_slice_3/stack:output:07neural_tensor_layer_12/strided_slice_3/stack_1:output:07neural_tensor_layer_12/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2(
&neural_tensor_layer_12/strided_slice_3?
neural_tensor_layer_12/MatMul_3MatMulattention_12/transpose_2:y:0/neural_tensor_layer_12/strided_slice_3:output:0*
T0*
_output_shapes

:2!
neural_tensor_layer_12/MatMul_3?
neural_tensor_layer_12/mul_2Mulattention_12/transpose_5:y:0)neural_tensor_layer_12/MatMul_3:product:0*
T0*
_output_shapes

:2
neural_tensor_layer_12/mul_2?
+neural_tensor_layer_12/add_2/ReadVariableOpReadVariableOp2neural_tensor_layer_12_add_readvariableop_resource*
_output_shapes
:*
dtype02-
+neural_tensor_layer_12/add_2/ReadVariableOp?
neural_tensor_layer_12/add_2AddV2 neural_tensor_layer_12/mul_2:z:03neural_tensor_layer_12/add_2/ReadVariableOp:value:0*
T0*
_output_shapes

:2
neural_tensor_layer_12/add_2?
.neural_tensor_layer_12/Sum_2/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :20
.neural_tensor_layer_12/Sum_2/reduction_indices?
neural_tensor_layer_12/Sum_2Sum neural_tensor_layer_12/add_2:z:07neural_tensor_layer_12/Sum_2/reduction_indices:output:0*
T0*
_output_shapes
:2
neural_tensor_layer_12/Sum_2?
'neural_tensor_layer_12/ReadVariableOp_3ReadVariableOp.neural_tensor_layer_12_readvariableop_resource*"
_output_shapes
:*
dtype02)
'neural_tensor_layer_12/ReadVariableOp_3?
,neural_tensor_layer_12/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:2.
,neural_tensor_layer_12/strided_slice_4/stack?
.neural_tensor_layer_12/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.neural_tensor_layer_12/strided_slice_4/stack_1?
.neural_tensor_layer_12/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.neural_tensor_layer_12/strided_slice_4/stack_2?
&neural_tensor_layer_12/strided_slice_4StridedSlice/neural_tensor_layer_12/ReadVariableOp_3:value:05neural_tensor_layer_12/strided_slice_4/stack:output:07neural_tensor_layer_12/strided_slice_4/stack_1:output:07neural_tensor_layer_12/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2(
&neural_tensor_layer_12/strided_slice_4?
neural_tensor_layer_12/MatMul_4MatMulattention_12/transpose_2:y:0/neural_tensor_layer_12/strided_slice_4:output:0*
T0*
_output_shapes

:2!
neural_tensor_layer_12/MatMul_4?
neural_tensor_layer_12/mul_3Mulattention_12/transpose_5:y:0)neural_tensor_layer_12/MatMul_4:product:0*
T0*
_output_shapes

:2
neural_tensor_layer_12/mul_3?
+neural_tensor_layer_12/add_3/ReadVariableOpReadVariableOp2neural_tensor_layer_12_add_readvariableop_resource*
_output_shapes
:*
dtype02-
+neural_tensor_layer_12/add_3/ReadVariableOp?
neural_tensor_layer_12/add_3AddV2 neural_tensor_layer_12/mul_3:z:03neural_tensor_layer_12/add_3/ReadVariableOp:value:0*
T0*
_output_shapes

:2
neural_tensor_layer_12/add_3?
.neural_tensor_layer_12/Sum_3/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :20
.neural_tensor_layer_12/Sum_3/reduction_indices?
neural_tensor_layer_12/Sum_3Sum neural_tensor_layer_12/add_3:z:07neural_tensor_layer_12/Sum_3/reduction_indices:output:0*
T0*
_output_shapes
:2
neural_tensor_layer_12/Sum_3?
'neural_tensor_layer_12/ReadVariableOp_4ReadVariableOp.neural_tensor_layer_12_readvariableop_resource*"
_output_shapes
:*
dtype02)
'neural_tensor_layer_12/ReadVariableOp_4?
,neural_tensor_layer_12/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:2.
,neural_tensor_layer_12/strided_slice_5/stack?
.neural_tensor_layer_12/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.neural_tensor_layer_12/strided_slice_5/stack_1?
.neural_tensor_layer_12/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.neural_tensor_layer_12/strided_slice_5/stack_2?
&neural_tensor_layer_12/strided_slice_5StridedSlice/neural_tensor_layer_12/ReadVariableOp_4:value:05neural_tensor_layer_12/strided_slice_5/stack:output:07neural_tensor_layer_12/strided_slice_5/stack_1:output:07neural_tensor_layer_12/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2(
&neural_tensor_layer_12/strided_slice_5?
neural_tensor_layer_12/MatMul_5MatMulattention_12/transpose_2:y:0/neural_tensor_layer_12/strided_slice_5:output:0*
T0*
_output_shapes

:2!
neural_tensor_layer_12/MatMul_5?
neural_tensor_layer_12/mul_4Mulattention_12/transpose_5:y:0)neural_tensor_layer_12/MatMul_5:product:0*
T0*
_output_shapes

:2
neural_tensor_layer_12/mul_4?
+neural_tensor_layer_12/add_4/ReadVariableOpReadVariableOp2neural_tensor_layer_12_add_readvariableop_resource*
_output_shapes
:*
dtype02-
+neural_tensor_layer_12/add_4/ReadVariableOp?
neural_tensor_layer_12/add_4AddV2 neural_tensor_layer_12/mul_4:z:03neural_tensor_layer_12/add_4/ReadVariableOp:value:0*
T0*
_output_shapes

:2
neural_tensor_layer_12/add_4?
.neural_tensor_layer_12/Sum_4/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :20
.neural_tensor_layer_12/Sum_4/reduction_indices?
neural_tensor_layer_12/Sum_4Sum neural_tensor_layer_12/add_4:z:07neural_tensor_layer_12/Sum_4/reduction_indices:output:0*
T0*
_output_shapes
:2
neural_tensor_layer_12/Sum_4?
'neural_tensor_layer_12/ReadVariableOp_5ReadVariableOp.neural_tensor_layer_12_readvariableop_resource*"
_output_shapes
:*
dtype02)
'neural_tensor_layer_12/ReadVariableOp_5?
,neural_tensor_layer_12/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB:2.
,neural_tensor_layer_12/strided_slice_6/stack?
.neural_tensor_layer_12/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.neural_tensor_layer_12/strided_slice_6/stack_1?
.neural_tensor_layer_12/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.neural_tensor_layer_12/strided_slice_6/stack_2?
&neural_tensor_layer_12/strided_slice_6StridedSlice/neural_tensor_layer_12/ReadVariableOp_5:value:05neural_tensor_layer_12/strided_slice_6/stack:output:07neural_tensor_layer_12/strided_slice_6/stack_1:output:07neural_tensor_layer_12/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2(
&neural_tensor_layer_12/strided_slice_6?
neural_tensor_layer_12/MatMul_6MatMulattention_12/transpose_2:y:0/neural_tensor_layer_12/strided_slice_6:output:0*
T0*
_output_shapes

:2!
neural_tensor_layer_12/MatMul_6?
neural_tensor_layer_12/mul_5Mulattention_12/transpose_5:y:0)neural_tensor_layer_12/MatMul_6:product:0*
T0*
_output_shapes

:2
neural_tensor_layer_12/mul_5?
+neural_tensor_layer_12/add_5/ReadVariableOpReadVariableOp2neural_tensor_layer_12_add_readvariableop_resource*
_output_shapes
:*
dtype02-
+neural_tensor_layer_12/add_5/ReadVariableOp?
neural_tensor_layer_12/add_5AddV2 neural_tensor_layer_12/mul_5:z:03neural_tensor_layer_12/add_5/ReadVariableOp:value:0*
T0*
_output_shapes

:2
neural_tensor_layer_12/add_5?
.neural_tensor_layer_12/Sum_5/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :20
.neural_tensor_layer_12/Sum_5/reduction_indices?
neural_tensor_layer_12/Sum_5Sum neural_tensor_layer_12/add_5:z:07neural_tensor_layer_12/Sum_5/reduction_indices:output:0*
T0*
_output_shapes
:2
neural_tensor_layer_12/Sum_5?
'neural_tensor_layer_12/ReadVariableOp_6ReadVariableOp.neural_tensor_layer_12_readvariableop_resource*"
_output_shapes
:*
dtype02)
'neural_tensor_layer_12/ReadVariableOp_6?
,neural_tensor_layer_12/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB:2.
,neural_tensor_layer_12/strided_slice_7/stack?
.neural_tensor_layer_12/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.neural_tensor_layer_12/strided_slice_7/stack_1?
.neural_tensor_layer_12/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.neural_tensor_layer_12/strided_slice_7/stack_2?
&neural_tensor_layer_12/strided_slice_7StridedSlice/neural_tensor_layer_12/ReadVariableOp_6:value:05neural_tensor_layer_12/strided_slice_7/stack:output:07neural_tensor_layer_12/strided_slice_7/stack_1:output:07neural_tensor_layer_12/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2(
&neural_tensor_layer_12/strided_slice_7?
neural_tensor_layer_12/MatMul_7MatMulattention_12/transpose_2:y:0/neural_tensor_layer_12/strided_slice_7:output:0*
T0*
_output_shapes

:2!
neural_tensor_layer_12/MatMul_7?
neural_tensor_layer_12/mul_6Mulattention_12/transpose_5:y:0)neural_tensor_layer_12/MatMul_7:product:0*
T0*
_output_shapes

:2
neural_tensor_layer_12/mul_6?
+neural_tensor_layer_12/add_6/ReadVariableOpReadVariableOp2neural_tensor_layer_12_add_readvariableop_resource*
_output_shapes
:*
dtype02-
+neural_tensor_layer_12/add_6/ReadVariableOp?
neural_tensor_layer_12/add_6AddV2 neural_tensor_layer_12/mul_6:z:03neural_tensor_layer_12/add_6/ReadVariableOp:value:0*
T0*
_output_shapes

:2
neural_tensor_layer_12/add_6?
.neural_tensor_layer_12/Sum_6/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :20
.neural_tensor_layer_12/Sum_6/reduction_indices?
neural_tensor_layer_12/Sum_6Sum neural_tensor_layer_12/add_6:z:07neural_tensor_layer_12/Sum_6/reduction_indices:output:0*
T0*
_output_shapes
:2
neural_tensor_layer_12/Sum_6?
'neural_tensor_layer_12/ReadVariableOp_7ReadVariableOp.neural_tensor_layer_12_readvariableop_resource*"
_output_shapes
:*
dtype02)
'neural_tensor_layer_12/ReadVariableOp_7?
,neural_tensor_layer_12/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB:2.
,neural_tensor_layer_12/strided_slice_8/stack?
.neural_tensor_layer_12/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.neural_tensor_layer_12/strided_slice_8/stack_1?
.neural_tensor_layer_12/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.neural_tensor_layer_12/strided_slice_8/stack_2?
&neural_tensor_layer_12/strided_slice_8StridedSlice/neural_tensor_layer_12/ReadVariableOp_7:value:05neural_tensor_layer_12/strided_slice_8/stack:output:07neural_tensor_layer_12/strided_slice_8/stack_1:output:07neural_tensor_layer_12/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2(
&neural_tensor_layer_12/strided_slice_8?
neural_tensor_layer_12/MatMul_8MatMulattention_12/transpose_2:y:0/neural_tensor_layer_12/strided_slice_8:output:0*
T0*
_output_shapes

:2!
neural_tensor_layer_12/MatMul_8?
neural_tensor_layer_12/mul_7Mulattention_12/transpose_5:y:0)neural_tensor_layer_12/MatMul_8:product:0*
T0*
_output_shapes

:2
neural_tensor_layer_12/mul_7?
+neural_tensor_layer_12/add_7/ReadVariableOpReadVariableOp2neural_tensor_layer_12_add_readvariableop_resource*
_output_shapes
:*
dtype02-
+neural_tensor_layer_12/add_7/ReadVariableOp?
neural_tensor_layer_12/add_7AddV2 neural_tensor_layer_12/mul_7:z:03neural_tensor_layer_12/add_7/ReadVariableOp:value:0*
T0*
_output_shapes

:2
neural_tensor_layer_12/add_7?
.neural_tensor_layer_12/Sum_7/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :20
.neural_tensor_layer_12/Sum_7/reduction_indices?
neural_tensor_layer_12/Sum_7Sum neural_tensor_layer_12/add_7:z:07neural_tensor_layer_12/Sum_7/reduction_indices:output:0*
T0*
_output_shapes
:2
neural_tensor_layer_12/Sum_7?
'neural_tensor_layer_12/ReadVariableOp_8ReadVariableOp.neural_tensor_layer_12_readvariableop_resource*"
_output_shapes
:*
dtype02)
'neural_tensor_layer_12/ReadVariableOp_8?
,neural_tensor_layer_12/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB:2.
,neural_tensor_layer_12/strided_slice_9/stack?
.neural_tensor_layer_12/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:	20
.neural_tensor_layer_12/strided_slice_9/stack_1?
.neural_tensor_layer_12/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.neural_tensor_layer_12/strided_slice_9/stack_2?
&neural_tensor_layer_12/strided_slice_9StridedSlice/neural_tensor_layer_12/ReadVariableOp_8:value:05neural_tensor_layer_12/strided_slice_9/stack:output:07neural_tensor_layer_12/strided_slice_9/stack_1:output:07neural_tensor_layer_12/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2(
&neural_tensor_layer_12/strided_slice_9?
neural_tensor_layer_12/MatMul_9MatMulattention_12/transpose_2:y:0/neural_tensor_layer_12/strided_slice_9:output:0*
T0*
_output_shapes

:2!
neural_tensor_layer_12/MatMul_9?
neural_tensor_layer_12/mul_8Mulattention_12/transpose_5:y:0)neural_tensor_layer_12/MatMul_9:product:0*
T0*
_output_shapes

:2
neural_tensor_layer_12/mul_8?
+neural_tensor_layer_12/add_8/ReadVariableOpReadVariableOp2neural_tensor_layer_12_add_readvariableop_resource*
_output_shapes
:*
dtype02-
+neural_tensor_layer_12/add_8/ReadVariableOp?
neural_tensor_layer_12/add_8AddV2 neural_tensor_layer_12/mul_8:z:03neural_tensor_layer_12/add_8/ReadVariableOp:value:0*
T0*
_output_shapes

:2
neural_tensor_layer_12/add_8?
.neural_tensor_layer_12/Sum_8/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :20
.neural_tensor_layer_12/Sum_8/reduction_indices?
neural_tensor_layer_12/Sum_8Sum neural_tensor_layer_12/add_8:z:07neural_tensor_layer_12/Sum_8/reduction_indices:output:0*
T0*
_output_shapes
:2
neural_tensor_layer_12/Sum_8?
'neural_tensor_layer_12/ReadVariableOp_9ReadVariableOp.neural_tensor_layer_12_readvariableop_resource*"
_output_shapes
:*
dtype02)
'neural_tensor_layer_12/ReadVariableOp_9?
-neural_tensor_layer_12/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB:	2/
-neural_tensor_layer_12/strided_slice_10/stack?
/neural_tensor_layer_12/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
21
/neural_tensor_layer_12/strided_slice_10/stack_1?
/neural_tensor_layer_12/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/neural_tensor_layer_12/strided_slice_10/stack_2?
'neural_tensor_layer_12/strided_slice_10StridedSlice/neural_tensor_layer_12/ReadVariableOp_9:value:06neural_tensor_layer_12/strided_slice_10/stack:output:08neural_tensor_layer_12/strided_slice_10/stack_1:output:08neural_tensor_layer_12/strided_slice_10/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2)
'neural_tensor_layer_12/strided_slice_10?
 neural_tensor_layer_12/MatMul_10MatMulattention_12/transpose_2:y:00neural_tensor_layer_12/strided_slice_10:output:0*
T0*
_output_shapes

:2"
 neural_tensor_layer_12/MatMul_10?
neural_tensor_layer_12/mul_9Mulattention_12/transpose_5:y:0*neural_tensor_layer_12/MatMul_10:product:0*
T0*
_output_shapes

:2
neural_tensor_layer_12/mul_9?
+neural_tensor_layer_12/add_9/ReadVariableOpReadVariableOp2neural_tensor_layer_12_add_readvariableop_resource*
_output_shapes
:*
dtype02-
+neural_tensor_layer_12/add_9/ReadVariableOp?
neural_tensor_layer_12/add_9AddV2 neural_tensor_layer_12/mul_9:z:03neural_tensor_layer_12/add_9/ReadVariableOp:value:0*
T0*
_output_shapes

:2
neural_tensor_layer_12/add_9?
.neural_tensor_layer_12/Sum_9/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :20
.neural_tensor_layer_12/Sum_9/reduction_indices?
neural_tensor_layer_12/Sum_9Sum neural_tensor_layer_12/add_9:z:07neural_tensor_layer_12/Sum_9/reduction_indices:output:0*
T0*
_output_shapes
:2
neural_tensor_layer_12/Sum_9?
(neural_tensor_layer_12/ReadVariableOp_10ReadVariableOp.neural_tensor_layer_12_readvariableop_resource*"
_output_shapes
:*
dtype02*
(neural_tensor_layer_12/ReadVariableOp_10?
-neural_tensor_layer_12/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:
2/
-neural_tensor_layer_12/strided_slice_11/stack?
/neural_tensor_layer_12/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/neural_tensor_layer_12/strided_slice_11/stack_1?
/neural_tensor_layer_12/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/neural_tensor_layer_12/strided_slice_11/stack_2?
'neural_tensor_layer_12/strided_slice_11StridedSlice0neural_tensor_layer_12/ReadVariableOp_10:value:06neural_tensor_layer_12/strided_slice_11/stack:output:08neural_tensor_layer_12/strided_slice_11/stack_1:output:08neural_tensor_layer_12/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2)
'neural_tensor_layer_12/strided_slice_11?
 neural_tensor_layer_12/MatMul_11MatMulattention_12/transpose_2:y:00neural_tensor_layer_12/strided_slice_11:output:0*
T0*
_output_shapes

:2"
 neural_tensor_layer_12/MatMul_11?
neural_tensor_layer_12/mul_10Mulattention_12/transpose_5:y:0*neural_tensor_layer_12/MatMul_11:product:0*
T0*
_output_shapes

:2
neural_tensor_layer_12/mul_10?
,neural_tensor_layer_12/add_10/ReadVariableOpReadVariableOp2neural_tensor_layer_12_add_readvariableop_resource*
_output_shapes
:*
dtype02.
,neural_tensor_layer_12/add_10/ReadVariableOp?
neural_tensor_layer_12/add_10AddV2!neural_tensor_layer_12/mul_10:z:04neural_tensor_layer_12/add_10/ReadVariableOp:value:0*
T0*
_output_shapes

:2
neural_tensor_layer_12/add_10?
/neural_tensor_layer_12/Sum_10/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :21
/neural_tensor_layer_12/Sum_10/reduction_indices?
neural_tensor_layer_12/Sum_10Sum!neural_tensor_layer_12/add_10:z:08neural_tensor_layer_12/Sum_10/reduction_indices:output:0*
T0*
_output_shapes
:2
neural_tensor_layer_12/Sum_10?
(neural_tensor_layer_12/ReadVariableOp_11ReadVariableOp.neural_tensor_layer_12_readvariableop_resource*"
_output_shapes
:*
dtype02*
(neural_tensor_layer_12/ReadVariableOp_11?
-neural_tensor_layer_12/strided_slice_12/stackConst*
_output_shapes
:*
dtype0*
valueB:2/
-neural_tensor_layer_12/strided_slice_12/stack?
/neural_tensor_layer_12/strided_slice_12/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/neural_tensor_layer_12/strided_slice_12/stack_1?
/neural_tensor_layer_12/strided_slice_12/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/neural_tensor_layer_12/strided_slice_12/stack_2?
'neural_tensor_layer_12/strided_slice_12StridedSlice0neural_tensor_layer_12/ReadVariableOp_11:value:06neural_tensor_layer_12/strided_slice_12/stack:output:08neural_tensor_layer_12/strided_slice_12/stack_1:output:08neural_tensor_layer_12/strided_slice_12/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2)
'neural_tensor_layer_12/strided_slice_12?
 neural_tensor_layer_12/MatMul_12MatMulattention_12/transpose_2:y:00neural_tensor_layer_12/strided_slice_12:output:0*
T0*
_output_shapes

:2"
 neural_tensor_layer_12/MatMul_12?
neural_tensor_layer_12/mul_11Mulattention_12/transpose_5:y:0*neural_tensor_layer_12/MatMul_12:product:0*
T0*
_output_shapes

:2
neural_tensor_layer_12/mul_11?
,neural_tensor_layer_12/add_11/ReadVariableOpReadVariableOp2neural_tensor_layer_12_add_readvariableop_resource*
_output_shapes
:*
dtype02.
,neural_tensor_layer_12/add_11/ReadVariableOp?
neural_tensor_layer_12/add_11AddV2!neural_tensor_layer_12/mul_11:z:04neural_tensor_layer_12/add_11/ReadVariableOp:value:0*
T0*
_output_shapes

:2
neural_tensor_layer_12/add_11?
/neural_tensor_layer_12/Sum_11/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :21
/neural_tensor_layer_12/Sum_11/reduction_indices?
neural_tensor_layer_12/Sum_11Sum!neural_tensor_layer_12/add_11:z:08neural_tensor_layer_12/Sum_11/reduction_indices:output:0*
T0*
_output_shapes
:2
neural_tensor_layer_12/Sum_11?
(neural_tensor_layer_12/ReadVariableOp_12ReadVariableOp.neural_tensor_layer_12_readvariableop_resource*"
_output_shapes
:*
dtype02*
(neural_tensor_layer_12/ReadVariableOp_12?
-neural_tensor_layer_12/strided_slice_13/stackConst*
_output_shapes
:*
dtype0*
valueB:2/
-neural_tensor_layer_12/strided_slice_13/stack?
/neural_tensor_layer_12/strided_slice_13/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/neural_tensor_layer_12/strided_slice_13/stack_1?
/neural_tensor_layer_12/strided_slice_13/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/neural_tensor_layer_12/strided_slice_13/stack_2?
'neural_tensor_layer_12/strided_slice_13StridedSlice0neural_tensor_layer_12/ReadVariableOp_12:value:06neural_tensor_layer_12/strided_slice_13/stack:output:08neural_tensor_layer_12/strided_slice_13/stack_1:output:08neural_tensor_layer_12/strided_slice_13/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2)
'neural_tensor_layer_12/strided_slice_13?
 neural_tensor_layer_12/MatMul_13MatMulattention_12/transpose_2:y:00neural_tensor_layer_12/strided_slice_13:output:0*
T0*
_output_shapes

:2"
 neural_tensor_layer_12/MatMul_13?
neural_tensor_layer_12/mul_12Mulattention_12/transpose_5:y:0*neural_tensor_layer_12/MatMul_13:product:0*
T0*
_output_shapes

:2
neural_tensor_layer_12/mul_12?
,neural_tensor_layer_12/add_12/ReadVariableOpReadVariableOp2neural_tensor_layer_12_add_readvariableop_resource*
_output_shapes
:*
dtype02.
,neural_tensor_layer_12/add_12/ReadVariableOp?
neural_tensor_layer_12/add_12AddV2!neural_tensor_layer_12/mul_12:z:04neural_tensor_layer_12/add_12/ReadVariableOp:value:0*
T0*
_output_shapes

:2
neural_tensor_layer_12/add_12?
/neural_tensor_layer_12/Sum_12/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :21
/neural_tensor_layer_12/Sum_12/reduction_indices?
neural_tensor_layer_12/Sum_12Sum!neural_tensor_layer_12/add_12:z:08neural_tensor_layer_12/Sum_12/reduction_indices:output:0*
T0*
_output_shapes
:2
neural_tensor_layer_12/Sum_12?
(neural_tensor_layer_12/ReadVariableOp_13ReadVariableOp.neural_tensor_layer_12_readvariableop_resource*"
_output_shapes
:*
dtype02*
(neural_tensor_layer_12/ReadVariableOp_13?
-neural_tensor_layer_12/strided_slice_14/stackConst*
_output_shapes
:*
dtype0*
valueB:2/
-neural_tensor_layer_12/strided_slice_14/stack?
/neural_tensor_layer_12/strided_slice_14/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/neural_tensor_layer_12/strided_slice_14/stack_1?
/neural_tensor_layer_12/strided_slice_14/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/neural_tensor_layer_12/strided_slice_14/stack_2?
'neural_tensor_layer_12/strided_slice_14StridedSlice0neural_tensor_layer_12/ReadVariableOp_13:value:06neural_tensor_layer_12/strided_slice_14/stack:output:08neural_tensor_layer_12/strided_slice_14/stack_1:output:08neural_tensor_layer_12/strided_slice_14/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2)
'neural_tensor_layer_12/strided_slice_14?
 neural_tensor_layer_12/MatMul_14MatMulattention_12/transpose_2:y:00neural_tensor_layer_12/strided_slice_14:output:0*
T0*
_output_shapes

:2"
 neural_tensor_layer_12/MatMul_14?
neural_tensor_layer_12/mul_13Mulattention_12/transpose_5:y:0*neural_tensor_layer_12/MatMul_14:product:0*
T0*
_output_shapes

:2
neural_tensor_layer_12/mul_13?
,neural_tensor_layer_12/add_13/ReadVariableOpReadVariableOp2neural_tensor_layer_12_add_readvariableop_resource*
_output_shapes
:*
dtype02.
,neural_tensor_layer_12/add_13/ReadVariableOp?
neural_tensor_layer_12/add_13AddV2!neural_tensor_layer_12/mul_13:z:04neural_tensor_layer_12/add_13/ReadVariableOp:value:0*
T0*
_output_shapes

:2
neural_tensor_layer_12/add_13?
/neural_tensor_layer_12/Sum_13/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :21
/neural_tensor_layer_12/Sum_13/reduction_indices?
neural_tensor_layer_12/Sum_13Sum!neural_tensor_layer_12/add_13:z:08neural_tensor_layer_12/Sum_13/reduction_indices:output:0*
T0*
_output_shapes
:2
neural_tensor_layer_12/Sum_13?
(neural_tensor_layer_12/ReadVariableOp_14ReadVariableOp.neural_tensor_layer_12_readvariableop_resource*"
_output_shapes
:*
dtype02*
(neural_tensor_layer_12/ReadVariableOp_14?
-neural_tensor_layer_12/strided_slice_15/stackConst*
_output_shapes
:*
dtype0*
valueB:2/
-neural_tensor_layer_12/strided_slice_15/stack?
/neural_tensor_layer_12/strided_slice_15/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/neural_tensor_layer_12/strided_slice_15/stack_1?
/neural_tensor_layer_12/strided_slice_15/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/neural_tensor_layer_12/strided_slice_15/stack_2?
'neural_tensor_layer_12/strided_slice_15StridedSlice0neural_tensor_layer_12/ReadVariableOp_14:value:06neural_tensor_layer_12/strided_slice_15/stack:output:08neural_tensor_layer_12/strided_slice_15/stack_1:output:08neural_tensor_layer_12/strided_slice_15/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2)
'neural_tensor_layer_12/strided_slice_15?
 neural_tensor_layer_12/MatMul_15MatMulattention_12/transpose_2:y:00neural_tensor_layer_12/strided_slice_15:output:0*
T0*
_output_shapes

:2"
 neural_tensor_layer_12/MatMul_15?
neural_tensor_layer_12/mul_14Mulattention_12/transpose_5:y:0*neural_tensor_layer_12/MatMul_15:product:0*
T0*
_output_shapes

:2
neural_tensor_layer_12/mul_14?
,neural_tensor_layer_12/add_14/ReadVariableOpReadVariableOp2neural_tensor_layer_12_add_readvariableop_resource*
_output_shapes
:*
dtype02.
,neural_tensor_layer_12/add_14/ReadVariableOp?
neural_tensor_layer_12/add_14AddV2!neural_tensor_layer_12/mul_14:z:04neural_tensor_layer_12/add_14/ReadVariableOp:value:0*
T0*
_output_shapes

:2
neural_tensor_layer_12/add_14?
/neural_tensor_layer_12/Sum_14/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :21
/neural_tensor_layer_12/Sum_14/reduction_indices?
neural_tensor_layer_12/Sum_14Sum!neural_tensor_layer_12/add_14:z:08neural_tensor_layer_12/Sum_14/reduction_indices:output:0*
T0*
_output_shapes
:2
neural_tensor_layer_12/Sum_14?
(neural_tensor_layer_12/ReadVariableOp_15ReadVariableOp.neural_tensor_layer_12_readvariableop_resource*"
_output_shapes
:*
dtype02*
(neural_tensor_layer_12/ReadVariableOp_15?
-neural_tensor_layer_12/strided_slice_16/stackConst*
_output_shapes
:*
dtype0*
valueB:2/
-neural_tensor_layer_12/strided_slice_16/stack?
/neural_tensor_layer_12/strided_slice_16/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/neural_tensor_layer_12/strided_slice_16/stack_1?
/neural_tensor_layer_12/strided_slice_16/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/neural_tensor_layer_12/strided_slice_16/stack_2?
'neural_tensor_layer_12/strided_slice_16StridedSlice0neural_tensor_layer_12/ReadVariableOp_15:value:06neural_tensor_layer_12/strided_slice_16/stack:output:08neural_tensor_layer_12/strided_slice_16/stack_1:output:08neural_tensor_layer_12/strided_slice_16/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2)
'neural_tensor_layer_12/strided_slice_16?
 neural_tensor_layer_12/MatMul_16MatMulattention_12/transpose_2:y:00neural_tensor_layer_12/strided_slice_16:output:0*
T0*
_output_shapes

:2"
 neural_tensor_layer_12/MatMul_16?
neural_tensor_layer_12/mul_15Mulattention_12/transpose_5:y:0*neural_tensor_layer_12/MatMul_16:product:0*
T0*
_output_shapes

:2
neural_tensor_layer_12/mul_15?
,neural_tensor_layer_12/add_15/ReadVariableOpReadVariableOp2neural_tensor_layer_12_add_readvariableop_resource*
_output_shapes
:*
dtype02.
,neural_tensor_layer_12/add_15/ReadVariableOp?
neural_tensor_layer_12/add_15AddV2!neural_tensor_layer_12/mul_15:z:04neural_tensor_layer_12/add_15/ReadVariableOp:value:0*
T0*
_output_shapes

:2
neural_tensor_layer_12/add_15?
/neural_tensor_layer_12/Sum_15/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :21
/neural_tensor_layer_12/Sum_15/reduction_indices?
neural_tensor_layer_12/Sum_15Sum!neural_tensor_layer_12/add_15:z:08neural_tensor_layer_12/Sum_15/reduction_indices:output:0*
T0*
_output_shapes
:2
neural_tensor_layer_12/Sum_15?
$neural_tensor_layer_12/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2&
$neural_tensor_layer_12/concat_1/axis?
neural_tensor_layer_12/concat_1ConcatV2#neural_tensor_layer_12/Sum:output:0%neural_tensor_layer_12/Sum_1:output:0%neural_tensor_layer_12/Sum_2:output:0%neural_tensor_layer_12/Sum_3:output:0%neural_tensor_layer_12/Sum_4:output:0%neural_tensor_layer_12/Sum_5:output:0%neural_tensor_layer_12/Sum_6:output:0%neural_tensor_layer_12/Sum_7:output:0%neural_tensor_layer_12/Sum_8:output:0%neural_tensor_layer_12/Sum_9:output:0&neural_tensor_layer_12/Sum_10:output:0&neural_tensor_layer_12/Sum_11:output:0&neural_tensor_layer_12/Sum_12:output:0&neural_tensor_layer_12/Sum_13:output:0&neural_tensor_layer_12/Sum_14:output:0&neural_tensor_layer_12/Sum_15:output:0-neural_tensor_layer_12/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2!
neural_tensor_layer_12/concat_1?
&neural_tensor_layer_12/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2(
&neural_tensor_layer_12/Reshape/shape/1?
$neural_tensor_layer_12/Reshape/shapePack-neural_tensor_layer_12/strided_slice:output:0/neural_tensor_layer_12/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2&
$neural_tensor_layer_12/Reshape/shape?
neural_tensor_layer_12/ReshapeReshape(neural_tensor_layer_12/concat_1:output:0-neural_tensor_layer_12/Reshape/shape:output:0*
T0*
_output_shapes

:2 
neural_tensor_layer_12/Reshape?
neural_tensor_layer_12/add_16AddV2'neural_tensor_layer_12/Reshape:output:0'neural_tensor_layer_12/MatMul:product:0*
T0*
_output_shapes

:2
neural_tensor_layer_12/add_16?
neural_tensor_layer_12/TanhTanh!neural_tensor_layer_12/add_16:z:0*
T0*
_output_shapes

:2
neural_tensor_layer_12/Tanh?
dense_48/MatMul/ReadVariableOpReadVariableOp'dense_48_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_48/MatMul/ReadVariableOp?
dense_48/MatMulMatMulneural_tensor_layer_12/Tanh:y:0&dense_48/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense_48/MatMul?
dense_48/BiasAdd/ReadVariableOpReadVariableOp(dense_48_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_48/BiasAdd/ReadVariableOp?
dense_48/BiasAddBiasAdddense_48/MatMul:product:0'dense_48/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense_48/BiasAddj
dense_48/ReluReludense_48/BiasAdd:output:0*
T0*
_output_shapes

:2
dense_48/Relu?
dense_49/MatMul/ReadVariableOpReadVariableOp'dense_49_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_49/MatMul/ReadVariableOp?
dense_49/MatMulMatMuldense_48/Relu:activations:0&dense_49/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense_49/MatMul?
dense_49/BiasAdd/ReadVariableOpReadVariableOp(dense_49_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_49/BiasAdd/ReadVariableOp?
dense_49/BiasAddBiasAdddense_49/MatMul:product:0'dense_49/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense_49/BiasAddj
dense_49/ReluReludense_49/BiasAdd:output:0*
T0*
_output_shapes

:2
dense_49/Relu?
dense_50/MatMul/ReadVariableOpReadVariableOp'dense_50_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_50/MatMul/ReadVariableOp?
dense_50/MatMulMatMuldense_49/Relu:activations:0&dense_50/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense_50/MatMul?
dense_50/BiasAdd/ReadVariableOpReadVariableOp(dense_50_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_50/BiasAdd/ReadVariableOp?
dense_50/BiasAddBiasAdddense_50/MatMul:product:0'dense_50/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense_50/BiasAddj
dense_50/ReluReludense_50/BiasAdd:output:0*
T0*
_output_shapes

:2
dense_50/Relu?
dense_51/MatMul/ReadVariableOpReadVariableOp'dense_51_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_51/MatMul/ReadVariableOp?
dense_51/MatMulMatMuldense_50/Relu:activations:0&dense_51/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense_51/MatMul?
dense_51/BiasAdd/ReadVariableOpReadVariableOp(dense_51_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_51/BiasAdd/ReadVariableOp?
dense_51/BiasAddBiasAdddense_51/MatMul:product:0'dense_51/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense_51/BiasAdd?
tf.math.sigmoid_12/SigmoidSigmoiddense_51/BiasAdd:output:0*
T0*
_output_shapes

:2
tf.math.sigmoid_12/Sigmoidp
IdentityIdentitytf.math.sigmoid_12/Sigmoid:y:0^NoOp*
T0*
_output_shapes

:2

Identity?
NoOpNoOp#^attention_12/MatMul/ReadVariableOp%^attention_12/MatMul_3/ReadVariableOp ^dense_48/BiasAdd/ReadVariableOp^dense_48/MatMul/ReadVariableOp ^dense_49/BiasAdd/ReadVariableOp^dense_49/MatMul/ReadVariableOp ^dense_50/BiasAdd/ReadVariableOp^dense_50/MatMul/ReadVariableOp ^dense_51/BiasAdd/ReadVariableOp^dense_51/MatMul/ReadVariableOp#^graph_conv_36/add_1/ReadVariableOp#^graph_conv_36/add_5/ReadVariableOp'^graph_conv_36/transpose/ReadVariableOp)^graph_conv_36/transpose_2/ReadVariableOp#^graph_conv_37/add_1/ReadVariableOp#^graph_conv_37/add_5/ReadVariableOp'^graph_conv_37/transpose/ReadVariableOp)^graph_conv_37/transpose_2/ReadVariableOp#^graph_conv_38/add_1/ReadVariableOp#^graph_conv_38/add_5/ReadVariableOp'^graph_conv_38/transpose/ReadVariableOp)^graph_conv_38/transpose_2/ReadVariableOp-^neural_tensor_layer_12/MatMul/ReadVariableOp&^neural_tensor_layer_12/ReadVariableOp(^neural_tensor_layer_12/ReadVariableOp_1)^neural_tensor_layer_12/ReadVariableOp_10)^neural_tensor_layer_12/ReadVariableOp_11)^neural_tensor_layer_12/ReadVariableOp_12)^neural_tensor_layer_12/ReadVariableOp_13)^neural_tensor_layer_12/ReadVariableOp_14)^neural_tensor_layer_12/ReadVariableOp_15(^neural_tensor_layer_12/ReadVariableOp_2(^neural_tensor_layer_12/ReadVariableOp_3(^neural_tensor_layer_12/ReadVariableOp_4(^neural_tensor_layer_12/ReadVariableOp_5(^neural_tensor_layer_12/ReadVariableOp_6(^neural_tensor_layer_12/ReadVariableOp_7(^neural_tensor_layer_12/ReadVariableOp_8(^neural_tensor_layer_12/ReadVariableOp_9*^neural_tensor_layer_12/add/ReadVariableOp,^neural_tensor_layer_12/add_1/ReadVariableOp-^neural_tensor_layer_12/add_10/ReadVariableOp-^neural_tensor_layer_12/add_11/ReadVariableOp-^neural_tensor_layer_12/add_12/ReadVariableOp-^neural_tensor_layer_12/add_13/ReadVariableOp-^neural_tensor_layer_12/add_14/ReadVariableOp-^neural_tensor_layer_12/add_15/ReadVariableOp,^neural_tensor_layer_12/add_2/ReadVariableOp,^neural_tensor_layer_12/add_3/ReadVariableOp,^neural_tensor_layer_12/add_4/ReadVariableOp,^neural_tensor_layer_12/add_5/ReadVariableOp,^neural_tensor_layer_12/add_6/ReadVariableOp,^neural_tensor_layer_12/add_7/ReadVariableOp,^neural_tensor_layer_12/add_8/ReadVariableOp,^neural_tensor_layer_12/add_9/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????	:'???????????????????????????:?????????	:'???????????????????????????: : : : : : : : : : : : : : : : : : 2H
"attention_12/MatMul/ReadVariableOp"attention_12/MatMul/ReadVariableOp2L
$attention_12/MatMul_3/ReadVariableOp$attention_12/MatMul_3/ReadVariableOp2B
dense_48/BiasAdd/ReadVariableOpdense_48/BiasAdd/ReadVariableOp2@
dense_48/MatMul/ReadVariableOpdense_48/MatMul/ReadVariableOp2B
dense_49/BiasAdd/ReadVariableOpdense_49/BiasAdd/ReadVariableOp2@
dense_49/MatMul/ReadVariableOpdense_49/MatMul/ReadVariableOp2B
dense_50/BiasAdd/ReadVariableOpdense_50/BiasAdd/ReadVariableOp2@
dense_50/MatMul/ReadVariableOpdense_50/MatMul/ReadVariableOp2B
dense_51/BiasAdd/ReadVariableOpdense_51/BiasAdd/ReadVariableOp2@
dense_51/MatMul/ReadVariableOpdense_51/MatMul/ReadVariableOp2H
"graph_conv_36/add_1/ReadVariableOp"graph_conv_36/add_1/ReadVariableOp2H
"graph_conv_36/add_5/ReadVariableOp"graph_conv_36/add_5/ReadVariableOp2P
&graph_conv_36/transpose/ReadVariableOp&graph_conv_36/transpose/ReadVariableOp2T
(graph_conv_36/transpose_2/ReadVariableOp(graph_conv_36/transpose_2/ReadVariableOp2H
"graph_conv_37/add_1/ReadVariableOp"graph_conv_37/add_1/ReadVariableOp2H
"graph_conv_37/add_5/ReadVariableOp"graph_conv_37/add_5/ReadVariableOp2P
&graph_conv_37/transpose/ReadVariableOp&graph_conv_37/transpose/ReadVariableOp2T
(graph_conv_37/transpose_2/ReadVariableOp(graph_conv_37/transpose_2/ReadVariableOp2H
"graph_conv_38/add_1/ReadVariableOp"graph_conv_38/add_1/ReadVariableOp2H
"graph_conv_38/add_5/ReadVariableOp"graph_conv_38/add_5/ReadVariableOp2P
&graph_conv_38/transpose/ReadVariableOp&graph_conv_38/transpose/ReadVariableOp2T
(graph_conv_38/transpose_2/ReadVariableOp(graph_conv_38/transpose_2/ReadVariableOp2\
,neural_tensor_layer_12/MatMul/ReadVariableOp,neural_tensor_layer_12/MatMul/ReadVariableOp2N
%neural_tensor_layer_12/ReadVariableOp%neural_tensor_layer_12/ReadVariableOp2R
'neural_tensor_layer_12/ReadVariableOp_1'neural_tensor_layer_12/ReadVariableOp_12T
(neural_tensor_layer_12/ReadVariableOp_10(neural_tensor_layer_12/ReadVariableOp_102T
(neural_tensor_layer_12/ReadVariableOp_11(neural_tensor_layer_12/ReadVariableOp_112T
(neural_tensor_layer_12/ReadVariableOp_12(neural_tensor_layer_12/ReadVariableOp_122T
(neural_tensor_layer_12/ReadVariableOp_13(neural_tensor_layer_12/ReadVariableOp_132T
(neural_tensor_layer_12/ReadVariableOp_14(neural_tensor_layer_12/ReadVariableOp_142T
(neural_tensor_layer_12/ReadVariableOp_15(neural_tensor_layer_12/ReadVariableOp_152R
'neural_tensor_layer_12/ReadVariableOp_2'neural_tensor_layer_12/ReadVariableOp_22R
'neural_tensor_layer_12/ReadVariableOp_3'neural_tensor_layer_12/ReadVariableOp_32R
'neural_tensor_layer_12/ReadVariableOp_4'neural_tensor_layer_12/ReadVariableOp_42R
'neural_tensor_layer_12/ReadVariableOp_5'neural_tensor_layer_12/ReadVariableOp_52R
'neural_tensor_layer_12/ReadVariableOp_6'neural_tensor_layer_12/ReadVariableOp_62R
'neural_tensor_layer_12/ReadVariableOp_7'neural_tensor_layer_12/ReadVariableOp_72R
'neural_tensor_layer_12/ReadVariableOp_8'neural_tensor_layer_12/ReadVariableOp_82R
'neural_tensor_layer_12/ReadVariableOp_9'neural_tensor_layer_12/ReadVariableOp_92V
)neural_tensor_layer_12/add/ReadVariableOp)neural_tensor_layer_12/add/ReadVariableOp2Z
+neural_tensor_layer_12/add_1/ReadVariableOp+neural_tensor_layer_12/add_1/ReadVariableOp2\
,neural_tensor_layer_12/add_10/ReadVariableOp,neural_tensor_layer_12/add_10/ReadVariableOp2\
,neural_tensor_layer_12/add_11/ReadVariableOp,neural_tensor_layer_12/add_11/ReadVariableOp2\
,neural_tensor_layer_12/add_12/ReadVariableOp,neural_tensor_layer_12/add_12/ReadVariableOp2\
,neural_tensor_layer_12/add_13/ReadVariableOp,neural_tensor_layer_12/add_13/ReadVariableOp2\
,neural_tensor_layer_12/add_14/ReadVariableOp,neural_tensor_layer_12/add_14/ReadVariableOp2\
,neural_tensor_layer_12/add_15/ReadVariableOp,neural_tensor_layer_12/add_15/ReadVariableOp2Z
+neural_tensor_layer_12/add_2/ReadVariableOp+neural_tensor_layer_12/add_2/ReadVariableOp2Z
+neural_tensor_layer_12/add_3/ReadVariableOp+neural_tensor_layer_12/add_3/ReadVariableOp2Z
+neural_tensor_layer_12/add_4/ReadVariableOp+neural_tensor_layer_12/add_4/ReadVariableOp2Z
+neural_tensor_layer_12/add_5/ReadVariableOp+neural_tensor_layer_12/add_5/ReadVariableOp2Z
+neural_tensor_layer_12/add_6/ReadVariableOp+neural_tensor_layer_12/add_6/ReadVariableOp2Z
+neural_tensor_layer_12/add_7/ReadVariableOp+neural_tensor_layer_12/add_7/ReadVariableOp2Z
+neural_tensor_layer_12/add_8/ReadVariableOp+neural_tensor_layer_12/add_8/ReadVariableOp2Z
+neural_tensor_layer_12/add_9/ReadVariableOp+neural_tensor_layer_12/add_9/ReadVariableOp:U Q
+
_output_shapes
:?????????	
"
_user_specified_name
inputs/0:gc
=
_output_shapes+
):'???????????????????????????
"
_user_specified_name
inputs/1:UQ
+
_output_shapes
:?????????	
"
_user_specified_name
inputs/2:gc
=
_output_shapes+
):'???????????????????????????
"
_user_specified_name
inputs/3
?
?
*__inference_model_12_layer_call_fn_4263530
input_49
input_50
input_51
input_52
unknown:@
	unknown_0:@
	unknown_1:@ 
	unknown_2: 
	unknown_3: 
	unknown_4:
	unknown_5:
	unknown_6: 
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:

unknown_16:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_49input_50input_51input_52unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*!
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_model_12_layer_call_and_return_conditional_losses_42634472
StatefulPartitionedCallr
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????	:'???????????????????????????:?????????	:'???????????????????????????: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:?????????	
"
_user_specified_name
input_49:gc
=
_output_shapes+
):'???????????????????????????
"
_user_specified_name
input_50:UQ
+
_output_shapes
:?????????	
"
_user_specified_name
input_51:gc
=
_output_shapes+
):'???????????????????????????
"
_user_specified_name
input_52
?
?
*__inference_model_12_layer_call_fn_4263812
inputs_0
inputs_1
inputs_2
inputs_3
unknown:@
	unknown_0:@
	unknown_1:@ 
	unknown_2: 
	unknown_3: 
	unknown_4:
	unknown_5:
	unknown_6: 
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:

unknown_16:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1inputs_2inputs_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*!
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_model_12_layer_call_and_return_conditional_losses_42634472
StatefulPartitionedCallr
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????	:'???????????????????????????:?????????	:'???????????????????????????: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:?????????	
"
_user_specified_name
inputs/0:gc
=
_output_shapes+
):'???????????????????????????
"
_user_specified_name
inputs/1:UQ
+
_output_shapes
:?????????	
"
_user_specified_name
inputs/2:gc
=
_output_shapes+
):'???????????????????????????
"
_user_specified_name
inputs/3
?
?
*__inference_model_12_layer_call_fn_4263768
inputs_0
inputs_1
inputs_2
inputs_3
unknown:@
	unknown_0:@
	unknown_1:@ 
	unknown_2: 
	unknown_3: 
	unknown_4:
	unknown_5:
	unknown_6: 
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:

unknown_16:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1inputs_2inputs_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*!
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_model_12_layer_call_and_return_conditional_losses_42630122
StatefulPartitionedCallr
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????	:'???????????????????????????:?????????	:'???????????????????????????: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:?????????	
"
_user_specified_name
inputs/0:gc
=
_output_shapes+
):'???????????????????????????
"
_user_specified_name
inputs/1:UQ
+
_output_shapes
:?????????	
"
_user_specified_name
inputs/2:gc
=
_output_shapes+
):'???????????????????????????
"
_user_specified_name
inputs/3
?,
?
J__inference_graph_conv_37_layer_call_and_return_conditional_losses_4262632

inputs
inputs_11
shape_2_readvariableop_resource:@ +
add_1_readvariableop_resource: 
identity??add_1/ReadVariableOp?transpose/ReadVariableOp}
MatMulBatchMatMulV2inputs_1inputs_1*
T0*=
_output_shapes+
):'???????????????????????????2
MatMulM
ShapeShapeMatMul:output:0*
T0*
_output_shapes
:2
Shapev
addAddV2MatMul:output:0inputs_1*
T0*=
_output_shapes+
):'???????????????????????????2
add[
	Greater/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
	Greater/y?
GreaterGreateradd:z:0Greater/y:output:0*
T0*=
_output_shapes+
):'???????????????????????????2	
Greaterx
CastCastGreater:z:0*

DstT0*

SrcT0
*=
_output_shapes+
):'???????????????????????????2
CastH
Shape_1Shapeinputs*
T0*
_output_shapes
:2	
Shape_1^
unstackUnpackShape_1:output:0*
T0*
_output_shapes
: : : *	
num2	
unstack?
Shape_2/ReadVariableOpReadVariableOpshape_2_readvariableop_resource*
_output_shapes

:@ *
dtype02
Shape_2/ReadVariableOpc
Shape_2Const*
_output_shapes
:*
dtype0*
valueB"@       2	
Shape_2`
	unstack_1UnpackShape_2:output:0*
T0*
_output_shapes
: : *	
num2
	unstack_1o
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   2
Reshape/shapeo
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:?????????@2	
Reshape?
transpose/ReadVariableOpReadVariableOpshape_2_readvariableop_resource*
_output_shapes

:@ *
dtype02
transpose/ReadVariableOpq
transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose/perm?
	transpose	Transpose transpose/ReadVariableOp:value:0transpose/perm:output:0*
T0*
_output_shapes

:@ 2
	transposes
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"@   ????2
Reshape_1/shapes
	Reshape_1Reshapetranspose:y:0Reshape_1/shape:output:0*
T0*
_output_shapes

:@ 2
	Reshape_1v
MatMul_1MatMulReshape:output:0Reshape_1:output:0*
T0*'
_output_shapes
:????????? 2

MatMul_1h
Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B : 2
Reshape_2/shape/2?
Reshape_2/shapePackunstack:output:0unstack:output:1Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape_2/shape?
	Reshape_2ReshapeMatMul_1:product:0Reshape_2/shape:output:0*
T0*4
_output_shapes"
 :?????????????????? 2
	Reshape_2?
add_1/ReadVariableOpReadVariableOpadd_1_readvariableop_resource*
_output_shapes
: *
dtype02
add_1/ReadVariableOp?
add_1AddV2Reshape_2:output:0add_1/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :?????????????????? 2
add_1?
MatMul_2BatchMatMulV2Cast:y:0Cast:y:0*
T0*=
_output_shapes+
):'???????????????????????????2

MatMul_2S
Shape_3ShapeMatMul_2:output:0*
T0*
_output_shapes
:2	
Shape_3|
add_2AddV2MatMul_2:output:0Cast:y:0*
T0*=
_output_shapes+
):'???????????????????????????2
add_2_
Greater_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
Greater_1/y?
	Greater_1Greater	add_2:z:0Greater_1/y:output:0*
T0*=
_output_shapes+
):'???????????????????????????2
	Greater_1~
Cast_1CastGreater_1:z:0*

DstT0*

SrcT0
*=
_output_shapes+
):'???????????????????????????2
Cast_1y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose
Cast_1:y:0transpose_1/perm:output:0*
T0*=
_output_shapes+
):'???????????????????????????2
transpose_1?
MatMul_3BatchMatMulV2transpose_1:y:0	add_1:z:0*
T0*4
_output_shapes"
 :?????????????????? 2

MatMul_3S
Shape_4ShapeMatMul_3:output:0*
T0*
_output_shapes
:2	
Shape_4p
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indices?
SumSum
Cast_1:y:0Sum/reduction_indices:output:0*
T0*4
_output_shapes"
 :??????????????????*
	keep_dims(2
SumW
add_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32	
add_3/yv
add_3AddV2Sum:output:0add_3/y:output:0*
T0*4
_output_shapes"
 :??????????????????2
add_3z
truedivRealDivMatMul_3:output:0	add_3:z:0*
T0*4
_output_shapes"
 :?????????????????? 2	
truediv`
ReluRelutruediv:z:0*
T0*4
_output_shapes"
 :?????????????????? 2
Reluz
IdentityIdentityRelu:activations:0^NoOp*
T0*4
_output_shapes"
 :?????????????????? 2

Identity?
NoOpNoOp^add_1/ReadVariableOp^transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*`
_input_shapesO
M:??????????????????@:'???????????????????????????: : 2,
add_1/ReadVariableOpadd_1/ReadVariableOp24
transpose/ReadVariableOptranspose/ReadVariableOp:\ X
4
_output_shapes"
 :??????????????????@
 
_user_specified_nameinputs:ea
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
?
I__inference_attention_12_layer_call_and_return_conditional_losses_4262729
	embedding0
matmul_readvariableop_resource:
identity??MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOpv
MatMulMatMul	embeddingMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMulr
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2
Mean/reduction_indicesl
MeanMeanMatMul:product:0Mean/reduction_indices:output:0*
T0*
_output_shapes
:2
MeanH
TanhTanhMean:output:0*
T0*
_output_shapes
:2
Tanho
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ????2
Reshape/shapeh
ReshapeReshapeTanh:y:0Reshape/shape:output:0*
T0*
_output_shapes

:2	
Reshapeq
transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose/permw
	transpose	TransposeReshape:output:0transpose/perm:output:0*
T0*
_output_shapes

:2
	transposej
MatMul_1MatMul	embeddingtranspose:y:0*
T0*'
_output_shapes
:?????????2

MatMul_1c
SigmoidSigmoidMatMul_1:product:0*
T0*'
_output_shapes
:?????????2	
Sigmoidu
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm
transpose_1	Transpose	embeddingtranspose_1/perm:output:0*
T0*'
_output_shapes
:?????????2
transpose_1e
MatMul_2MatMultranspose_1:y:0Sigmoid:y:0*
T0*
_output_shapes

:2

MatMul_2u
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_2/perm
transpose_2	TransposeMatMul_2:product:0transpose_2/perm:output:0*
T0*
_output_shapes

:2
transpose_2a
IdentityIdentitytranspose_2:y:0^NoOp*
T0*
_output_shapes

:2

Identityf
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:R N
'
_output_shapes
:?????????
#
_user_specified_name	embedding
?	
?
/__inference_graph_conv_38_layer_call_fn_4265148
inputs_0
inputs_1
unknown: 
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_graph_conv_38_layer_call_and_return_conditional_losses_42631762
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*`
_input_shapesO
M:?????????????????? :'???????????????????????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :?????????????????? 
"
_user_specified_name
inputs/0:gc
=
_output_shapes+
):'???????????????????????????
"
_user_specified_name
inputs/1
?

?
E__inference_dense_49_layer_call_and_return_conditional_losses_4265528

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOpj
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpx
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2	
BiasAddO
ReluReluBiasAdd:output:0*
T0*
_output_shapes

:2
Relud
IdentityIdentityRelu:activations:0^NoOp*
T0*
_output_shapes

:2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
:: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:F B

_output_shapes

:
 
_user_specified_nameinputs
?
?
.__inference_attention_12_layer_call_fn_4265255
	embedding
unknown:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall	embeddingunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_attention_12_layer_call_and_return_conditional_losses_42627292
StatefulPartitionedCallr
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: 22
StatefulPartitionedCallStatefulPartitionedCall:R N
'
_output_shapes
:?????????
#
_user_specified_name	embedding
ԗ
? 
"__inference__wrapped_model_4262509
input_49
input_50
input_51
input_52H
6model_12_graph_conv_36_shape_2_readvariableop_resource:@B
4model_12_graph_conv_36_add_1_readvariableop_resource:@H
6model_12_graph_conv_37_shape_2_readvariableop_resource:@ B
4model_12_graph_conv_37_add_1_readvariableop_resource: H
6model_12_graph_conv_38_shape_2_readvariableop_resource: B
4model_12_graph_conv_38_add_1_readvariableop_resource:F
4model_12_attention_12_matmul_readvariableop_resource:P
>model_12_neural_tensor_layer_12_matmul_readvariableop_resource: M
7model_12_neural_tensor_layer_12_readvariableop_resource:I
;model_12_neural_tensor_layer_12_add_readvariableop_resource:B
0model_12_dense_48_matmul_readvariableop_resource:?
1model_12_dense_48_biasadd_readvariableop_resource:B
0model_12_dense_49_matmul_readvariableop_resource:?
1model_12_dense_49_biasadd_readvariableop_resource:B
0model_12_dense_50_matmul_readvariableop_resource:?
1model_12_dense_50_biasadd_readvariableop_resource:B
0model_12_dense_51_matmul_readvariableop_resource:?
1model_12_dense_51_biasadd_readvariableop_resource:
identity??+model_12/attention_12/MatMul/ReadVariableOp?-model_12/attention_12/MatMul_3/ReadVariableOp?(model_12/dense_48/BiasAdd/ReadVariableOp?'model_12/dense_48/MatMul/ReadVariableOp?(model_12/dense_49/BiasAdd/ReadVariableOp?'model_12/dense_49/MatMul/ReadVariableOp?(model_12/dense_50/BiasAdd/ReadVariableOp?'model_12/dense_50/MatMul/ReadVariableOp?(model_12/dense_51/BiasAdd/ReadVariableOp?'model_12/dense_51/MatMul/ReadVariableOp?+model_12/graph_conv_36/add_1/ReadVariableOp?+model_12/graph_conv_36/add_5/ReadVariableOp?/model_12/graph_conv_36/transpose/ReadVariableOp?1model_12/graph_conv_36/transpose_2/ReadVariableOp?+model_12/graph_conv_37/add_1/ReadVariableOp?+model_12/graph_conv_37/add_5/ReadVariableOp?/model_12/graph_conv_37/transpose/ReadVariableOp?1model_12/graph_conv_37/transpose_2/ReadVariableOp?+model_12/graph_conv_38/add_1/ReadVariableOp?+model_12/graph_conv_38/add_5/ReadVariableOp?/model_12/graph_conv_38/transpose/ReadVariableOp?1model_12/graph_conv_38/transpose_2/ReadVariableOp?5model_12/neural_tensor_layer_12/MatMul/ReadVariableOp?.model_12/neural_tensor_layer_12/ReadVariableOp?0model_12/neural_tensor_layer_12/ReadVariableOp_1?1model_12/neural_tensor_layer_12/ReadVariableOp_10?1model_12/neural_tensor_layer_12/ReadVariableOp_11?1model_12/neural_tensor_layer_12/ReadVariableOp_12?1model_12/neural_tensor_layer_12/ReadVariableOp_13?1model_12/neural_tensor_layer_12/ReadVariableOp_14?1model_12/neural_tensor_layer_12/ReadVariableOp_15?0model_12/neural_tensor_layer_12/ReadVariableOp_2?0model_12/neural_tensor_layer_12/ReadVariableOp_3?0model_12/neural_tensor_layer_12/ReadVariableOp_4?0model_12/neural_tensor_layer_12/ReadVariableOp_5?0model_12/neural_tensor_layer_12/ReadVariableOp_6?0model_12/neural_tensor_layer_12/ReadVariableOp_7?0model_12/neural_tensor_layer_12/ReadVariableOp_8?0model_12/neural_tensor_layer_12/ReadVariableOp_9?2model_12/neural_tensor_layer_12/add/ReadVariableOp?4model_12/neural_tensor_layer_12/add_1/ReadVariableOp?5model_12/neural_tensor_layer_12/add_10/ReadVariableOp?5model_12/neural_tensor_layer_12/add_11/ReadVariableOp?5model_12/neural_tensor_layer_12/add_12/ReadVariableOp?5model_12/neural_tensor_layer_12/add_13/ReadVariableOp?5model_12/neural_tensor_layer_12/add_14/ReadVariableOp?5model_12/neural_tensor_layer_12/add_15/ReadVariableOp?4model_12/neural_tensor_layer_12/add_2/ReadVariableOp?4model_12/neural_tensor_layer_12/add_3/ReadVariableOp?4model_12/neural_tensor_layer_12/add_4/ReadVariableOp?4model_12/neural_tensor_layer_12/add_5/ReadVariableOp?4model_12/neural_tensor_layer_12/add_6/ReadVariableOp?4model_12/neural_tensor_layer_12/add_7/ReadVariableOp?4model_12/neural_tensor_layer_12/add_8/ReadVariableOp?4model_12/neural_tensor_layer_12/add_9/ReadVariableOp?
model_12/graph_conv_36/MatMulBatchMatMulV2input_52input_52*
T0*=
_output_shapes+
):'???????????????????????????2
model_12/graph_conv_36/MatMul?
model_12/graph_conv_36/ShapeShape&model_12/graph_conv_36/MatMul:output:0*
T0*
_output_shapes
:2
model_12/graph_conv_36/Shape?
model_12/graph_conv_36/addAddV2&model_12/graph_conv_36/MatMul:output:0input_52*
T0*=
_output_shapes+
):'???????????????????????????2
model_12/graph_conv_36/add?
 model_12/graph_conv_36/Greater/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 model_12/graph_conv_36/Greater/y?
model_12/graph_conv_36/GreaterGreatermodel_12/graph_conv_36/add:z:0)model_12/graph_conv_36/Greater/y:output:0*
T0*=
_output_shapes+
):'???????????????????????????2 
model_12/graph_conv_36/Greater?
model_12/graph_conv_36/CastCast"model_12/graph_conv_36/Greater:z:0*

DstT0*

SrcT0
*=
_output_shapes+
):'???????????????????????????2
model_12/graph_conv_36/Castx
model_12/graph_conv_36/Shape_1Shapeinput_51*
T0*
_output_shapes
:2 
model_12/graph_conv_36/Shape_1?
model_12/graph_conv_36/unstackUnpack'model_12/graph_conv_36/Shape_1:output:0*
T0*
_output_shapes
: : : *	
num2 
model_12/graph_conv_36/unstack?
-model_12/graph_conv_36/Shape_2/ReadVariableOpReadVariableOp6model_12_graph_conv_36_shape_2_readvariableop_resource*
_output_shapes

:@*
dtype02/
-model_12/graph_conv_36/Shape_2/ReadVariableOp?
model_12/graph_conv_36/Shape_2Const*
_output_shapes
:*
dtype0*
valueB"   @   2 
model_12/graph_conv_36/Shape_2?
 model_12/graph_conv_36/unstack_1Unpack'model_12/graph_conv_36/Shape_2:output:0*
T0*
_output_shapes
: : *	
num2"
 model_12/graph_conv_36/unstack_1?
$model_12/graph_conv_36/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2&
$model_12/graph_conv_36/Reshape/shape?
model_12/graph_conv_36/ReshapeReshapeinput_51-model_12/graph_conv_36/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2 
model_12/graph_conv_36/Reshape?
/model_12/graph_conv_36/transpose/ReadVariableOpReadVariableOp6model_12_graph_conv_36_shape_2_readvariableop_resource*
_output_shapes

:@*
dtype021
/model_12/graph_conv_36/transpose/ReadVariableOp?
%model_12/graph_conv_36/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2'
%model_12/graph_conv_36/transpose/perm?
 model_12/graph_conv_36/transpose	Transpose7model_12/graph_conv_36/transpose/ReadVariableOp:value:0.model_12/graph_conv_36/transpose/perm:output:0*
T0*
_output_shapes

:@2"
 model_12/graph_conv_36/transpose?
&model_12/graph_conv_36/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ????2(
&model_12/graph_conv_36/Reshape_1/shape?
 model_12/graph_conv_36/Reshape_1Reshape$model_12/graph_conv_36/transpose:y:0/model_12/graph_conv_36/Reshape_1/shape:output:0*
T0*
_output_shapes

:@2"
 model_12/graph_conv_36/Reshape_1?
model_12/graph_conv_36/MatMul_1MatMul'model_12/graph_conv_36/Reshape:output:0)model_12/graph_conv_36/Reshape_1:output:0*
T0*'
_output_shapes
:?????????@2!
model_12/graph_conv_36/MatMul_1?
(model_12/graph_conv_36/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :	2*
(model_12/graph_conv_36/Reshape_2/shape/1?
(model_12/graph_conv_36/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :@2*
(model_12/graph_conv_36/Reshape_2/shape/2?
&model_12/graph_conv_36/Reshape_2/shapePack'model_12/graph_conv_36/unstack:output:01model_12/graph_conv_36/Reshape_2/shape/1:output:01model_12/graph_conv_36/Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:2(
&model_12/graph_conv_36/Reshape_2/shape?
 model_12/graph_conv_36/Reshape_2Reshape)model_12/graph_conv_36/MatMul_1:product:0/model_12/graph_conv_36/Reshape_2/shape:output:0*
T0*+
_output_shapes
:?????????	@2"
 model_12/graph_conv_36/Reshape_2?
+model_12/graph_conv_36/add_1/ReadVariableOpReadVariableOp4model_12_graph_conv_36_add_1_readvariableop_resource*
_output_shapes
:@*
dtype02-
+model_12/graph_conv_36/add_1/ReadVariableOp?
model_12/graph_conv_36/add_1AddV2)model_12/graph_conv_36/Reshape_2:output:03model_12/graph_conv_36/add_1/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????	@2
model_12/graph_conv_36/add_1?
model_12/graph_conv_36/MatMul_2BatchMatMulV2model_12/graph_conv_36/Cast:y:0model_12/graph_conv_36/Cast:y:0*
T0*=
_output_shapes+
):'???????????????????????????2!
model_12/graph_conv_36/MatMul_2?
model_12/graph_conv_36/Shape_3Shape(model_12/graph_conv_36/MatMul_2:output:0*
T0*
_output_shapes
:2 
model_12/graph_conv_36/Shape_3?
model_12/graph_conv_36/add_2AddV2(model_12/graph_conv_36/MatMul_2:output:0model_12/graph_conv_36/Cast:y:0*
T0*=
_output_shapes+
):'???????????????????????????2
model_12/graph_conv_36/add_2?
"model_12/graph_conv_36/Greater_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"model_12/graph_conv_36/Greater_1/y?
 model_12/graph_conv_36/Greater_1Greater model_12/graph_conv_36/add_2:z:0+model_12/graph_conv_36/Greater_1/y:output:0*
T0*=
_output_shapes+
):'???????????????????????????2"
 model_12/graph_conv_36/Greater_1?
model_12/graph_conv_36/Cast_1Cast$model_12/graph_conv_36/Greater_1:z:0*

DstT0*

SrcT0
*=
_output_shapes+
):'???????????????????????????2
model_12/graph_conv_36/Cast_1?
'model_12/graph_conv_36/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2)
'model_12/graph_conv_36/transpose_1/perm?
"model_12/graph_conv_36/transpose_1	Transpose!model_12/graph_conv_36/Cast_1:y:00model_12/graph_conv_36/transpose_1/perm:output:0*
T0*=
_output_shapes+
):'???????????????????????????2$
"model_12/graph_conv_36/transpose_1?
model_12/graph_conv_36/MatMul_3BatchMatMulV2&model_12/graph_conv_36/transpose_1:y:0 model_12/graph_conv_36/add_1:z:0*
T0*4
_output_shapes"
 :??????????????????@2!
model_12/graph_conv_36/MatMul_3?
model_12/graph_conv_36/Shape_4Shape(model_12/graph_conv_36/MatMul_3:output:0*
T0*
_output_shapes
:2 
model_12/graph_conv_36/Shape_4?
,model_12/graph_conv_36/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2.
,model_12/graph_conv_36/Sum/reduction_indices?
model_12/graph_conv_36/SumSum!model_12/graph_conv_36/Cast_1:y:05model_12/graph_conv_36/Sum/reduction_indices:output:0*
T0*4
_output_shapes"
 :??????????????????*
	keep_dims(2
model_12/graph_conv_36/Sum?
model_12/graph_conv_36/add_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32 
model_12/graph_conv_36/add_3/y?
model_12/graph_conv_36/add_3AddV2#model_12/graph_conv_36/Sum:output:0'model_12/graph_conv_36/add_3/y:output:0*
T0*4
_output_shapes"
 :??????????????????2
model_12/graph_conv_36/add_3?
model_12/graph_conv_36/truedivRealDiv(model_12/graph_conv_36/MatMul_3:output:0 model_12/graph_conv_36/add_3:z:0*
T0*4
_output_shapes"
 :??????????????????@2 
model_12/graph_conv_36/truediv?
model_12/graph_conv_36/ReluRelu"model_12/graph_conv_36/truediv:z:0*
T0*4
_output_shapes"
 :??????????????????@2
model_12/graph_conv_36/Relu?
model_12/graph_conv_36/MatMul_4BatchMatMulV2input_50input_50*
T0*=
_output_shapes+
):'???????????????????????????2!
model_12/graph_conv_36/MatMul_4?
model_12/graph_conv_36/Shape_5Shape(model_12/graph_conv_36/MatMul_4:output:0*
T0*
_output_shapes
:2 
model_12/graph_conv_36/Shape_5?
model_12/graph_conv_36/add_4AddV2(model_12/graph_conv_36/MatMul_4:output:0input_50*
T0*=
_output_shapes+
):'???????????????????????????2
model_12/graph_conv_36/add_4?
"model_12/graph_conv_36/Greater_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"model_12/graph_conv_36/Greater_2/y?
 model_12/graph_conv_36/Greater_2Greater model_12/graph_conv_36/add_4:z:0+model_12/graph_conv_36/Greater_2/y:output:0*
T0*=
_output_shapes+
):'???????????????????????????2"
 model_12/graph_conv_36/Greater_2?
model_12/graph_conv_36/Cast_2Cast$model_12/graph_conv_36/Greater_2:z:0*

DstT0*

SrcT0
*=
_output_shapes+
):'???????????????????????????2
model_12/graph_conv_36/Cast_2x
model_12/graph_conv_36/Shape_6Shapeinput_49*
T0*
_output_shapes
:2 
model_12/graph_conv_36/Shape_6?
 model_12/graph_conv_36/unstack_2Unpack'model_12/graph_conv_36/Shape_6:output:0*
T0*
_output_shapes
: : : *	
num2"
 model_12/graph_conv_36/unstack_2?
-model_12/graph_conv_36/Shape_7/ReadVariableOpReadVariableOp6model_12_graph_conv_36_shape_2_readvariableop_resource*
_output_shapes

:@*
dtype02/
-model_12/graph_conv_36/Shape_7/ReadVariableOp?
model_12/graph_conv_36/Shape_7Const*
_output_shapes
:*
dtype0*
valueB"   @   2 
model_12/graph_conv_36/Shape_7?
 model_12/graph_conv_36/unstack_3Unpack'model_12/graph_conv_36/Shape_7:output:0*
T0*
_output_shapes
: : *	
num2"
 model_12/graph_conv_36/unstack_3?
&model_12/graph_conv_36/Reshape_3/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2(
&model_12/graph_conv_36/Reshape_3/shape?
 model_12/graph_conv_36/Reshape_3Reshapeinput_49/model_12/graph_conv_36/Reshape_3/shape:output:0*
T0*'
_output_shapes
:?????????2"
 model_12/graph_conv_36/Reshape_3?
1model_12/graph_conv_36/transpose_2/ReadVariableOpReadVariableOp6model_12_graph_conv_36_shape_2_readvariableop_resource*
_output_shapes

:@*
dtype023
1model_12/graph_conv_36/transpose_2/ReadVariableOp?
'model_12/graph_conv_36/transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2)
'model_12/graph_conv_36/transpose_2/perm?
"model_12/graph_conv_36/transpose_2	Transpose9model_12/graph_conv_36/transpose_2/ReadVariableOp:value:00model_12/graph_conv_36/transpose_2/perm:output:0*
T0*
_output_shapes

:@2$
"model_12/graph_conv_36/transpose_2?
&model_12/graph_conv_36/Reshape_4/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ????2(
&model_12/graph_conv_36/Reshape_4/shape?
 model_12/graph_conv_36/Reshape_4Reshape&model_12/graph_conv_36/transpose_2:y:0/model_12/graph_conv_36/Reshape_4/shape:output:0*
T0*
_output_shapes

:@2"
 model_12/graph_conv_36/Reshape_4?
model_12/graph_conv_36/MatMul_5MatMul)model_12/graph_conv_36/Reshape_3:output:0)model_12/graph_conv_36/Reshape_4:output:0*
T0*'
_output_shapes
:?????????@2!
model_12/graph_conv_36/MatMul_5?
(model_12/graph_conv_36/Reshape_5/shape/1Const*
_output_shapes
: *
dtype0*
value	B :	2*
(model_12/graph_conv_36/Reshape_5/shape/1?
(model_12/graph_conv_36/Reshape_5/shape/2Const*
_output_shapes
: *
dtype0*
value	B :@2*
(model_12/graph_conv_36/Reshape_5/shape/2?
&model_12/graph_conv_36/Reshape_5/shapePack)model_12/graph_conv_36/unstack_2:output:01model_12/graph_conv_36/Reshape_5/shape/1:output:01model_12/graph_conv_36/Reshape_5/shape/2:output:0*
N*
T0*
_output_shapes
:2(
&model_12/graph_conv_36/Reshape_5/shape?
 model_12/graph_conv_36/Reshape_5Reshape)model_12/graph_conv_36/MatMul_5:product:0/model_12/graph_conv_36/Reshape_5/shape:output:0*
T0*+
_output_shapes
:?????????	@2"
 model_12/graph_conv_36/Reshape_5?
+model_12/graph_conv_36/add_5/ReadVariableOpReadVariableOp4model_12_graph_conv_36_add_1_readvariableop_resource*
_output_shapes
:@*
dtype02-
+model_12/graph_conv_36/add_5/ReadVariableOp?
model_12/graph_conv_36/add_5AddV2)model_12/graph_conv_36/Reshape_5:output:03model_12/graph_conv_36/add_5/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????	@2
model_12/graph_conv_36/add_5?
model_12/graph_conv_36/MatMul_6BatchMatMulV2!model_12/graph_conv_36/Cast_2:y:0!model_12/graph_conv_36/Cast_2:y:0*
T0*=
_output_shapes+
):'???????????????????????????2!
model_12/graph_conv_36/MatMul_6?
model_12/graph_conv_36/Shape_8Shape(model_12/graph_conv_36/MatMul_6:output:0*
T0*
_output_shapes
:2 
model_12/graph_conv_36/Shape_8?
model_12/graph_conv_36/add_6AddV2(model_12/graph_conv_36/MatMul_6:output:0!model_12/graph_conv_36/Cast_2:y:0*
T0*=
_output_shapes+
):'???????????????????????????2
model_12/graph_conv_36/add_6?
"model_12/graph_conv_36/Greater_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"model_12/graph_conv_36/Greater_3/y?
 model_12/graph_conv_36/Greater_3Greater model_12/graph_conv_36/add_6:z:0+model_12/graph_conv_36/Greater_3/y:output:0*
T0*=
_output_shapes+
):'???????????????????????????2"
 model_12/graph_conv_36/Greater_3?
model_12/graph_conv_36/Cast_3Cast$model_12/graph_conv_36/Greater_3:z:0*

DstT0*

SrcT0
*=
_output_shapes+
):'???????????????????????????2
model_12/graph_conv_36/Cast_3?
'model_12/graph_conv_36/transpose_3/permConst*
_output_shapes
:*
dtype0*!
valueB"          2)
'model_12/graph_conv_36/transpose_3/perm?
"model_12/graph_conv_36/transpose_3	Transpose!model_12/graph_conv_36/Cast_3:y:00model_12/graph_conv_36/transpose_3/perm:output:0*
T0*=
_output_shapes+
):'???????????????????????????2$
"model_12/graph_conv_36/transpose_3?
model_12/graph_conv_36/MatMul_7BatchMatMulV2&model_12/graph_conv_36/transpose_3:y:0 model_12/graph_conv_36/add_5:z:0*
T0*4
_output_shapes"
 :??????????????????@2!
model_12/graph_conv_36/MatMul_7?
model_12/graph_conv_36/Shape_9Shape(model_12/graph_conv_36/MatMul_7:output:0*
T0*
_output_shapes
:2 
model_12/graph_conv_36/Shape_9?
.model_12/graph_conv_36/Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :20
.model_12/graph_conv_36/Sum_1/reduction_indices?
model_12/graph_conv_36/Sum_1Sum!model_12/graph_conv_36/Cast_3:y:07model_12/graph_conv_36/Sum_1/reduction_indices:output:0*
T0*4
_output_shapes"
 :??????????????????*
	keep_dims(2
model_12/graph_conv_36/Sum_1?
model_12/graph_conv_36/add_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32 
model_12/graph_conv_36/add_7/y?
model_12/graph_conv_36/add_7AddV2%model_12/graph_conv_36/Sum_1:output:0'model_12/graph_conv_36/add_7/y:output:0*
T0*4
_output_shapes"
 :??????????????????2
model_12/graph_conv_36/add_7?
 model_12/graph_conv_36/truediv_1RealDiv(model_12/graph_conv_36/MatMul_7:output:0 model_12/graph_conv_36/add_7:z:0*
T0*4
_output_shapes"
 :??????????????????@2"
 model_12/graph_conv_36/truediv_1?
model_12/graph_conv_36/Relu_1Relu$model_12/graph_conv_36/truediv_1:z:0*
T0*4
_output_shapes"
 :??????????????????@2
model_12/graph_conv_36/Relu_1?
model_12/graph_conv_37/MatMulBatchMatMulV2input_52input_52*
T0*=
_output_shapes+
):'???????????????????????????2
model_12/graph_conv_37/MatMul?
model_12/graph_conv_37/ShapeShape&model_12/graph_conv_37/MatMul:output:0*
T0*
_output_shapes
:2
model_12/graph_conv_37/Shape?
model_12/graph_conv_37/addAddV2&model_12/graph_conv_37/MatMul:output:0input_52*
T0*=
_output_shapes+
):'???????????????????????????2
model_12/graph_conv_37/add?
 model_12/graph_conv_37/Greater/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 model_12/graph_conv_37/Greater/y?
model_12/graph_conv_37/GreaterGreatermodel_12/graph_conv_37/add:z:0)model_12/graph_conv_37/Greater/y:output:0*
T0*=
_output_shapes+
):'???????????????????????????2 
model_12/graph_conv_37/Greater?
model_12/graph_conv_37/CastCast"model_12/graph_conv_37/Greater:z:0*

DstT0*

SrcT0
*=
_output_shapes+
):'???????????????????????????2
model_12/graph_conv_37/Cast?
model_12/graph_conv_37/Shape_1Shape)model_12/graph_conv_36/Relu:activations:0*
T0*
_output_shapes
:2 
model_12/graph_conv_37/Shape_1?
model_12/graph_conv_37/unstackUnpack'model_12/graph_conv_37/Shape_1:output:0*
T0*
_output_shapes
: : : *	
num2 
model_12/graph_conv_37/unstack?
-model_12/graph_conv_37/Shape_2/ReadVariableOpReadVariableOp6model_12_graph_conv_37_shape_2_readvariableop_resource*
_output_shapes

:@ *
dtype02/
-model_12/graph_conv_37/Shape_2/ReadVariableOp?
model_12/graph_conv_37/Shape_2Const*
_output_shapes
:*
dtype0*
valueB"@       2 
model_12/graph_conv_37/Shape_2?
 model_12/graph_conv_37/unstack_1Unpack'model_12/graph_conv_37/Shape_2:output:0*
T0*
_output_shapes
: : *	
num2"
 model_12/graph_conv_37/unstack_1?
$model_12/graph_conv_37/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   2&
$model_12/graph_conv_37/Reshape/shape?
model_12/graph_conv_37/ReshapeReshape)model_12/graph_conv_36/Relu:activations:0-model_12/graph_conv_37/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????@2 
model_12/graph_conv_37/Reshape?
/model_12/graph_conv_37/transpose/ReadVariableOpReadVariableOp6model_12_graph_conv_37_shape_2_readvariableop_resource*
_output_shapes

:@ *
dtype021
/model_12/graph_conv_37/transpose/ReadVariableOp?
%model_12/graph_conv_37/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2'
%model_12/graph_conv_37/transpose/perm?
 model_12/graph_conv_37/transpose	Transpose7model_12/graph_conv_37/transpose/ReadVariableOp:value:0.model_12/graph_conv_37/transpose/perm:output:0*
T0*
_output_shapes

:@ 2"
 model_12/graph_conv_37/transpose?
&model_12/graph_conv_37/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"@   ????2(
&model_12/graph_conv_37/Reshape_1/shape?
 model_12/graph_conv_37/Reshape_1Reshape$model_12/graph_conv_37/transpose:y:0/model_12/graph_conv_37/Reshape_1/shape:output:0*
T0*
_output_shapes

:@ 2"
 model_12/graph_conv_37/Reshape_1?
model_12/graph_conv_37/MatMul_1MatMul'model_12/graph_conv_37/Reshape:output:0)model_12/graph_conv_37/Reshape_1:output:0*
T0*'
_output_shapes
:????????? 2!
model_12/graph_conv_37/MatMul_1?
(model_12/graph_conv_37/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B : 2*
(model_12/graph_conv_37/Reshape_2/shape/2?
&model_12/graph_conv_37/Reshape_2/shapePack'model_12/graph_conv_37/unstack:output:0'model_12/graph_conv_37/unstack:output:11model_12/graph_conv_37/Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:2(
&model_12/graph_conv_37/Reshape_2/shape?
 model_12/graph_conv_37/Reshape_2Reshape)model_12/graph_conv_37/MatMul_1:product:0/model_12/graph_conv_37/Reshape_2/shape:output:0*
T0*4
_output_shapes"
 :?????????????????? 2"
 model_12/graph_conv_37/Reshape_2?
+model_12/graph_conv_37/add_1/ReadVariableOpReadVariableOp4model_12_graph_conv_37_add_1_readvariableop_resource*
_output_shapes
: *
dtype02-
+model_12/graph_conv_37/add_1/ReadVariableOp?
model_12/graph_conv_37/add_1AddV2)model_12/graph_conv_37/Reshape_2:output:03model_12/graph_conv_37/add_1/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :?????????????????? 2
model_12/graph_conv_37/add_1?
model_12/graph_conv_37/MatMul_2BatchMatMulV2model_12/graph_conv_37/Cast:y:0model_12/graph_conv_37/Cast:y:0*
T0*=
_output_shapes+
):'???????????????????????????2!
model_12/graph_conv_37/MatMul_2?
model_12/graph_conv_37/Shape_3Shape(model_12/graph_conv_37/MatMul_2:output:0*
T0*
_output_shapes
:2 
model_12/graph_conv_37/Shape_3?
model_12/graph_conv_37/add_2AddV2(model_12/graph_conv_37/MatMul_2:output:0model_12/graph_conv_37/Cast:y:0*
T0*=
_output_shapes+
):'???????????????????????????2
model_12/graph_conv_37/add_2?
"model_12/graph_conv_37/Greater_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"model_12/graph_conv_37/Greater_1/y?
 model_12/graph_conv_37/Greater_1Greater model_12/graph_conv_37/add_2:z:0+model_12/graph_conv_37/Greater_1/y:output:0*
T0*=
_output_shapes+
):'???????????????????????????2"
 model_12/graph_conv_37/Greater_1?
model_12/graph_conv_37/Cast_1Cast$model_12/graph_conv_37/Greater_1:z:0*

DstT0*

SrcT0
*=
_output_shapes+
):'???????????????????????????2
model_12/graph_conv_37/Cast_1?
'model_12/graph_conv_37/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2)
'model_12/graph_conv_37/transpose_1/perm?
"model_12/graph_conv_37/transpose_1	Transpose!model_12/graph_conv_37/Cast_1:y:00model_12/graph_conv_37/transpose_1/perm:output:0*
T0*=
_output_shapes+
):'???????????????????????????2$
"model_12/graph_conv_37/transpose_1?
model_12/graph_conv_37/MatMul_3BatchMatMulV2&model_12/graph_conv_37/transpose_1:y:0 model_12/graph_conv_37/add_1:z:0*
T0*4
_output_shapes"
 :?????????????????? 2!
model_12/graph_conv_37/MatMul_3?
model_12/graph_conv_37/Shape_4Shape(model_12/graph_conv_37/MatMul_3:output:0*
T0*
_output_shapes
:2 
model_12/graph_conv_37/Shape_4?
,model_12/graph_conv_37/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2.
,model_12/graph_conv_37/Sum/reduction_indices?
model_12/graph_conv_37/SumSum!model_12/graph_conv_37/Cast_1:y:05model_12/graph_conv_37/Sum/reduction_indices:output:0*
T0*4
_output_shapes"
 :??????????????????*
	keep_dims(2
model_12/graph_conv_37/Sum?
model_12/graph_conv_37/add_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32 
model_12/graph_conv_37/add_3/y?
model_12/graph_conv_37/add_3AddV2#model_12/graph_conv_37/Sum:output:0'model_12/graph_conv_37/add_3/y:output:0*
T0*4
_output_shapes"
 :??????????????????2
model_12/graph_conv_37/add_3?
model_12/graph_conv_37/truedivRealDiv(model_12/graph_conv_37/MatMul_3:output:0 model_12/graph_conv_37/add_3:z:0*
T0*4
_output_shapes"
 :?????????????????? 2 
model_12/graph_conv_37/truediv?
model_12/graph_conv_37/ReluRelu"model_12/graph_conv_37/truediv:z:0*
T0*4
_output_shapes"
 :?????????????????? 2
model_12/graph_conv_37/Relu?
model_12/graph_conv_37/MatMul_4BatchMatMulV2input_50input_50*
T0*=
_output_shapes+
):'???????????????????????????2!
model_12/graph_conv_37/MatMul_4?
model_12/graph_conv_37/Shape_5Shape(model_12/graph_conv_37/MatMul_4:output:0*
T0*
_output_shapes
:2 
model_12/graph_conv_37/Shape_5?
model_12/graph_conv_37/add_4AddV2(model_12/graph_conv_37/MatMul_4:output:0input_50*
T0*=
_output_shapes+
):'???????????????????????????2
model_12/graph_conv_37/add_4?
"model_12/graph_conv_37/Greater_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"model_12/graph_conv_37/Greater_2/y?
 model_12/graph_conv_37/Greater_2Greater model_12/graph_conv_37/add_4:z:0+model_12/graph_conv_37/Greater_2/y:output:0*
T0*=
_output_shapes+
):'???????????????????????????2"
 model_12/graph_conv_37/Greater_2?
model_12/graph_conv_37/Cast_2Cast$model_12/graph_conv_37/Greater_2:z:0*

DstT0*

SrcT0
*=
_output_shapes+
):'???????????????????????????2
model_12/graph_conv_37/Cast_2?
model_12/graph_conv_37/Shape_6Shape+model_12/graph_conv_36/Relu_1:activations:0*
T0*
_output_shapes
:2 
model_12/graph_conv_37/Shape_6?
 model_12/graph_conv_37/unstack_2Unpack'model_12/graph_conv_37/Shape_6:output:0*
T0*
_output_shapes
: : : *	
num2"
 model_12/graph_conv_37/unstack_2?
-model_12/graph_conv_37/Shape_7/ReadVariableOpReadVariableOp6model_12_graph_conv_37_shape_2_readvariableop_resource*
_output_shapes

:@ *
dtype02/
-model_12/graph_conv_37/Shape_7/ReadVariableOp?
model_12/graph_conv_37/Shape_7Const*
_output_shapes
:*
dtype0*
valueB"@       2 
model_12/graph_conv_37/Shape_7?
 model_12/graph_conv_37/unstack_3Unpack'model_12/graph_conv_37/Shape_7:output:0*
T0*
_output_shapes
: : *	
num2"
 model_12/graph_conv_37/unstack_3?
&model_12/graph_conv_37/Reshape_3/shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   2(
&model_12/graph_conv_37/Reshape_3/shape?
 model_12/graph_conv_37/Reshape_3Reshape+model_12/graph_conv_36/Relu_1:activations:0/model_12/graph_conv_37/Reshape_3/shape:output:0*
T0*'
_output_shapes
:?????????@2"
 model_12/graph_conv_37/Reshape_3?
1model_12/graph_conv_37/transpose_2/ReadVariableOpReadVariableOp6model_12_graph_conv_37_shape_2_readvariableop_resource*
_output_shapes

:@ *
dtype023
1model_12/graph_conv_37/transpose_2/ReadVariableOp?
'model_12/graph_conv_37/transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2)
'model_12/graph_conv_37/transpose_2/perm?
"model_12/graph_conv_37/transpose_2	Transpose9model_12/graph_conv_37/transpose_2/ReadVariableOp:value:00model_12/graph_conv_37/transpose_2/perm:output:0*
T0*
_output_shapes

:@ 2$
"model_12/graph_conv_37/transpose_2?
&model_12/graph_conv_37/Reshape_4/shapeConst*
_output_shapes
:*
dtype0*
valueB"@   ????2(
&model_12/graph_conv_37/Reshape_4/shape?
 model_12/graph_conv_37/Reshape_4Reshape&model_12/graph_conv_37/transpose_2:y:0/model_12/graph_conv_37/Reshape_4/shape:output:0*
T0*
_output_shapes

:@ 2"
 model_12/graph_conv_37/Reshape_4?
model_12/graph_conv_37/MatMul_5MatMul)model_12/graph_conv_37/Reshape_3:output:0)model_12/graph_conv_37/Reshape_4:output:0*
T0*'
_output_shapes
:????????? 2!
model_12/graph_conv_37/MatMul_5?
(model_12/graph_conv_37/Reshape_5/shape/2Const*
_output_shapes
: *
dtype0*
value	B : 2*
(model_12/graph_conv_37/Reshape_5/shape/2?
&model_12/graph_conv_37/Reshape_5/shapePack)model_12/graph_conv_37/unstack_2:output:0)model_12/graph_conv_37/unstack_2:output:11model_12/graph_conv_37/Reshape_5/shape/2:output:0*
N*
T0*
_output_shapes
:2(
&model_12/graph_conv_37/Reshape_5/shape?
 model_12/graph_conv_37/Reshape_5Reshape)model_12/graph_conv_37/MatMul_5:product:0/model_12/graph_conv_37/Reshape_5/shape:output:0*
T0*4
_output_shapes"
 :?????????????????? 2"
 model_12/graph_conv_37/Reshape_5?
+model_12/graph_conv_37/add_5/ReadVariableOpReadVariableOp4model_12_graph_conv_37_add_1_readvariableop_resource*
_output_shapes
: *
dtype02-
+model_12/graph_conv_37/add_5/ReadVariableOp?
model_12/graph_conv_37/add_5AddV2)model_12/graph_conv_37/Reshape_5:output:03model_12/graph_conv_37/add_5/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :?????????????????? 2
model_12/graph_conv_37/add_5?
model_12/graph_conv_37/MatMul_6BatchMatMulV2!model_12/graph_conv_37/Cast_2:y:0!model_12/graph_conv_37/Cast_2:y:0*
T0*=
_output_shapes+
):'???????????????????????????2!
model_12/graph_conv_37/MatMul_6?
model_12/graph_conv_37/Shape_8Shape(model_12/graph_conv_37/MatMul_6:output:0*
T0*
_output_shapes
:2 
model_12/graph_conv_37/Shape_8?
model_12/graph_conv_37/add_6AddV2(model_12/graph_conv_37/MatMul_6:output:0!model_12/graph_conv_37/Cast_2:y:0*
T0*=
_output_shapes+
):'???????????????????????????2
model_12/graph_conv_37/add_6?
"model_12/graph_conv_37/Greater_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"model_12/graph_conv_37/Greater_3/y?
 model_12/graph_conv_37/Greater_3Greater model_12/graph_conv_37/add_6:z:0+model_12/graph_conv_37/Greater_3/y:output:0*
T0*=
_output_shapes+
):'???????????????????????????2"
 model_12/graph_conv_37/Greater_3?
model_12/graph_conv_37/Cast_3Cast$model_12/graph_conv_37/Greater_3:z:0*

DstT0*

SrcT0
*=
_output_shapes+
):'???????????????????????????2
model_12/graph_conv_37/Cast_3?
'model_12/graph_conv_37/transpose_3/permConst*
_output_shapes
:*
dtype0*!
valueB"          2)
'model_12/graph_conv_37/transpose_3/perm?
"model_12/graph_conv_37/transpose_3	Transpose!model_12/graph_conv_37/Cast_3:y:00model_12/graph_conv_37/transpose_3/perm:output:0*
T0*=
_output_shapes+
):'???????????????????????????2$
"model_12/graph_conv_37/transpose_3?
model_12/graph_conv_37/MatMul_7BatchMatMulV2&model_12/graph_conv_37/transpose_3:y:0 model_12/graph_conv_37/add_5:z:0*
T0*4
_output_shapes"
 :?????????????????? 2!
model_12/graph_conv_37/MatMul_7?
model_12/graph_conv_37/Shape_9Shape(model_12/graph_conv_37/MatMul_7:output:0*
T0*
_output_shapes
:2 
model_12/graph_conv_37/Shape_9?
.model_12/graph_conv_37/Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :20
.model_12/graph_conv_37/Sum_1/reduction_indices?
model_12/graph_conv_37/Sum_1Sum!model_12/graph_conv_37/Cast_3:y:07model_12/graph_conv_37/Sum_1/reduction_indices:output:0*
T0*4
_output_shapes"
 :??????????????????*
	keep_dims(2
model_12/graph_conv_37/Sum_1?
model_12/graph_conv_37/add_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32 
model_12/graph_conv_37/add_7/y?
model_12/graph_conv_37/add_7AddV2%model_12/graph_conv_37/Sum_1:output:0'model_12/graph_conv_37/add_7/y:output:0*
T0*4
_output_shapes"
 :??????????????????2
model_12/graph_conv_37/add_7?
 model_12/graph_conv_37/truediv_1RealDiv(model_12/graph_conv_37/MatMul_7:output:0 model_12/graph_conv_37/add_7:z:0*
T0*4
_output_shapes"
 :?????????????????? 2"
 model_12/graph_conv_37/truediv_1?
model_12/graph_conv_37/Relu_1Relu$model_12/graph_conv_37/truediv_1:z:0*
T0*4
_output_shapes"
 :?????????????????? 2
model_12/graph_conv_37/Relu_1?
model_12/graph_conv_38/MatMulBatchMatMulV2input_52input_52*
T0*=
_output_shapes+
):'???????????????????????????2
model_12/graph_conv_38/MatMul?
model_12/graph_conv_38/ShapeShape&model_12/graph_conv_38/MatMul:output:0*
T0*
_output_shapes
:2
model_12/graph_conv_38/Shape?
model_12/graph_conv_38/addAddV2&model_12/graph_conv_38/MatMul:output:0input_52*
T0*=
_output_shapes+
):'???????????????????????????2
model_12/graph_conv_38/add?
 model_12/graph_conv_38/Greater/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 model_12/graph_conv_38/Greater/y?
model_12/graph_conv_38/GreaterGreatermodel_12/graph_conv_38/add:z:0)model_12/graph_conv_38/Greater/y:output:0*
T0*=
_output_shapes+
):'???????????????????????????2 
model_12/graph_conv_38/Greater?
model_12/graph_conv_38/CastCast"model_12/graph_conv_38/Greater:z:0*

DstT0*

SrcT0
*=
_output_shapes+
):'???????????????????????????2
model_12/graph_conv_38/Cast?
model_12/graph_conv_38/Shape_1Shape)model_12/graph_conv_37/Relu:activations:0*
T0*
_output_shapes
:2 
model_12/graph_conv_38/Shape_1?
model_12/graph_conv_38/unstackUnpack'model_12/graph_conv_38/Shape_1:output:0*
T0*
_output_shapes
: : : *	
num2 
model_12/graph_conv_38/unstack?
-model_12/graph_conv_38/Shape_2/ReadVariableOpReadVariableOp6model_12_graph_conv_38_shape_2_readvariableop_resource*
_output_shapes

: *
dtype02/
-model_12/graph_conv_38/Shape_2/ReadVariableOp?
model_12/graph_conv_38/Shape_2Const*
_output_shapes
:*
dtype0*
valueB"       2 
model_12/graph_conv_38/Shape_2?
 model_12/graph_conv_38/unstack_1Unpack'model_12/graph_conv_38/Shape_2:output:0*
T0*
_output_shapes
: : *	
num2"
 model_12/graph_conv_38/unstack_1?
$model_12/graph_conv_38/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????    2&
$model_12/graph_conv_38/Reshape/shape?
model_12/graph_conv_38/ReshapeReshape)model_12/graph_conv_37/Relu:activations:0-model_12/graph_conv_38/Reshape/shape:output:0*
T0*'
_output_shapes
:????????? 2 
model_12/graph_conv_38/Reshape?
/model_12/graph_conv_38/transpose/ReadVariableOpReadVariableOp6model_12_graph_conv_38_shape_2_readvariableop_resource*
_output_shapes

: *
dtype021
/model_12/graph_conv_38/transpose/ReadVariableOp?
%model_12/graph_conv_38/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2'
%model_12/graph_conv_38/transpose/perm?
 model_12/graph_conv_38/transpose	Transpose7model_12/graph_conv_38/transpose/ReadVariableOp:value:0.model_12/graph_conv_38/transpose/perm:output:0*
T0*
_output_shapes

: 2"
 model_12/graph_conv_38/transpose?
&model_12/graph_conv_38/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"    ????2(
&model_12/graph_conv_38/Reshape_1/shape?
 model_12/graph_conv_38/Reshape_1Reshape$model_12/graph_conv_38/transpose:y:0/model_12/graph_conv_38/Reshape_1/shape:output:0*
T0*
_output_shapes

: 2"
 model_12/graph_conv_38/Reshape_1?
model_12/graph_conv_38/MatMul_1MatMul'model_12/graph_conv_38/Reshape:output:0)model_12/graph_conv_38/Reshape_1:output:0*
T0*'
_output_shapes
:?????????2!
model_12/graph_conv_38/MatMul_1?
(model_12/graph_conv_38/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2*
(model_12/graph_conv_38/Reshape_2/shape/2?
&model_12/graph_conv_38/Reshape_2/shapePack'model_12/graph_conv_38/unstack:output:0'model_12/graph_conv_38/unstack:output:11model_12/graph_conv_38/Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:2(
&model_12/graph_conv_38/Reshape_2/shape?
 model_12/graph_conv_38/Reshape_2Reshape)model_12/graph_conv_38/MatMul_1:product:0/model_12/graph_conv_38/Reshape_2/shape:output:0*
T0*4
_output_shapes"
 :??????????????????2"
 model_12/graph_conv_38/Reshape_2?
+model_12/graph_conv_38/add_1/ReadVariableOpReadVariableOp4model_12_graph_conv_38_add_1_readvariableop_resource*
_output_shapes
:*
dtype02-
+model_12/graph_conv_38/add_1/ReadVariableOp?
model_12/graph_conv_38/add_1AddV2)model_12/graph_conv_38/Reshape_2:output:03model_12/graph_conv_38/add_1/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :??????????????????2
model_12/graph_conv_38/add_1?
model_12/graph_conv_38/MatMul_2BatchMatMulV2model_12/graph_conv_38/Cast:y:0model_12/graph_conv_38/Cast:y:0*
T0*=
_output_shapes+
):'???????????????????????????2!
model_12/graph_conv_38/MatMul_2?
model_12/graph_conv_38/Shape_3Shape(model_12/graph_conv_38/MatMul_2:output:0*
T0*
_output_shapes
:2 
model_12/graph_conv_38/Shape_3?
model_12/graph_conv_38/add_2AddV2(model_12/graph_conv_38/MatMul_2:output:0model_12/graph_conv_38/Cast:y:0*
T0*=
_output_shapes+
):'???????????????????????????2
model_12/graph_conv_38/add_2?
"model_12/graph_conv_38/Greater_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"model_12/graph_conv_38/Greater_1/y?
 model_12/graph_conv_38/Greater_1Greater model_12/graph_conv_38/add_2:z:0+model_12/graph_conv_38/Greater_1/y:output:0*
T0*=
_output_shapes+
):'???????????????????????????2"
 model_12/graph_conv_38/Greater_1?
model_12/graph_conv_38/Cast_1Cast$model_12/graph_conv_38/Greater_1:z:0*

DstT0*

SrcT0
*=
_output_shapes+
):'???????????????????????????2
model_12/graph_conv_38/Cast_1?
'model_12/graph_conv_38/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2)
'model_12/graph_conv_38/transpose_1/perm?
"model_12/graph_conv_38/transpose_1	Transpose!model_12/graph_conv_38/Cast_1:y:00model_12/graph_conv_38/transpose_1/perm:output:0*
T0*=
_output_shapes+
):'???????????????????????????2$
"model_12/graph_conv_38/transpose_1?
model_12/graph_conv_38/MatMul_3BatchMatMulV2&model_12/graph_conv_38/transpose_1:y:0 model_12/graph_conv_38/add_1:z:0*
T0*4
_output_shapes"
 :??????????????????2!
model_12/graph_conv_38/MatMul_3?
model_12/graph_conv_38/Shape_4Shape(model_12/graph_conv_38/MatMul_3:output:0*
T0*
_output_shapes
:2 
model_12/graph_conv_38/Shape_4?
,model_12/graph_conv_38/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2.
,model_12/graph_conv_38/Sum/reduction_indices?
model_12/graph_conv_38/SumSum!model_12/graph_conv_38/Cast_1:y:05model_12/graph_conv_38/Sum/reduction_indices:output:0*
T0*4
_output_shapes"
 :??????????????????*
	keep_dims(2
model_12/graph_conv_38/Sum?
model_12/graph_conv_38/add_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32 
model_12/graph_conv_38/add_3/y?
model_12/graph_conv_38/add_3AddV2#model_12/graph_conv_38/Sum:output:0'model_12/graph_conv_38/add_3/y:output:0*
T0*4
_output_shapes"
 :??????????????????2
model_12/graph_conv_38/add_3?
model_12/graph_conv_38/truedivRealDiv(model_12/graph_conv_38/MatMul_3:output:0 model_12/graph_conv_38/add_3:z:0*
T0*4
_output_shapes"
 :??????????????????2 
model_12/graph_conv_38/truediv?
model_12/graph_conv_38/ReluRelu"model_12/graph_conv_38/truediv:z:0*
T0*4
_output_shapes"
 :??????????????????2
model_12/graph_conv_38/Relu?
model_12/graph_conv_38/MatMul_4BatchMatMulV2input_50input_50*
T0*=
_output_shapes+
):'???????????????????????????2!
model_12/graph_conv_38/MatMul_4?
model_12/graph_conv_38/Shape_5Shape(model_12/graph_conv_38/MatMul_4:output:0*
T0*
_output_shapes
:2 
model_12/graph_conv_38/Shape_5?
model_12/graph_conv_38/add_4AddV2(model_12/graph_conv_38/MatMul_4:output:0input_50*
T0*=
_output_shapes+
):'???????????????????????????2
model_12/graph_conv_38/add_4?
"model_12/graph_conv_38/Greater_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"model_12/graph_conv_38/Greater_2/y?
 model_12/graph_conv_38/Greater_2Greater model_12/graph_conv_38/add_4:z:0+model_12/graph_conv_38/Greater_2/y:output:0*
T0*=
_output_shapes+
):'???????????????????????????2"
 model_12/graph_conv_38/Greater_2?
model_12/graph_conv_38/Cast_2Cast$model_12/graph_conv_38/Greater_2:z:0*

DstT0*

SrcT0
*=
_output_shapes+
):'???????????????????????????2
model_12/graph_conv_38/Cast_2?
model_12/graph_conv_38/Shape_6Shape+model_12/graph_conv_37/Relu_1:activations:0*
T0*
_output_shapes
:2 
model_12/graph_conv_38/Shape_6?
 model_12/graph_conv_38/unstack_2Unpack'model_12/graph_conv_38/Shape_6:output:0*
T0*
_output_shapes
: : : *	
num2"
 model_12/graph_conv_38/unstack_2?
-model_12/graph_conv_38/Shape_7/ReadVariableOpReadVariableOp6model_12_graph_conv_38_shape_2_readvariableop_resource*
_output_shapes

: *
dtype02/
-model_12/graph_conv_38/Shape_7/ReadVariableOp?
model_12/graph_conv_38/Shape_7Const*
_output_shapes
:*
dtype0*
valueB"       2 
model_12/graph_conv_38/Shape_7?
 model_12/graph_conv_38/unstack_3Unpack'model_12/graph_conv_38/Shape_7:output:0*
T0*
_output_shapes
: : *	
num2"
 model_12/graph_conv_38/unstack_3?
&model_12/graph_conv_38/Reshape_3/shapeConst*
_output_shapes
:*
dtype0*
valueB"????    2(
&model_12/graph_conv_38/Reshape_3/shape?
 model_12/graph_conv_38/Reshape_3Reshape+model_12/graph_conv_37/Relu_1:activations:0/model_12/graph_conv_38/Reshape_3/shape:output:0*
T0*'
_output_shapes
:????????? 2"
 model_12/graph_conv_38/Reshape_3?
1model_12/graph_conv_38/transpose_2/ReadVariableOpReadVariableOp6model_12_graph_conv_38_shape_2_readvariableop_resource*
_output_shapes

: *
dtype023
1model_12/graph_conv_38/transpose_2/ReadVariableOp?
'model_12/graph_conv_38/transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2)
'model_12/graph_conv_38/transpose_2/perm?
"model_12/graph_conv_38/transpose_2	Transpose9model_12/graph_conv_38/transpose_2/ReadVariableOp:value:00model_12/graph_conv_38/transpose_2/perm:output:0*
T0*
_output_shapes

: 2$
"model_12/graph_conv_38/transpose_2?
&model_12/graph_conv_38/Reshape_4/shapeConst*
_output_shapes
:*
dtype0*
valueB"    ????2(
&model_12/graph_conv_38/Reshape_4/shape?
 model_12/graph_conv_38/Reshape_4Reshape&model_12/graph_conv_38/transpose_2:y:0/model_12/graph_conv_38/Reshape_4/shape:output:0*
T0*
_output_shapes

: 2"
 model_12/graph_conv_38/Reshape_4?
model_12/graph_conv_38/MatMul_5MatMul)model_12/graph_conv_38/Reshape_3:output:0)model_12/graph_conv_38/Reshape_4:output:0*
T0*'
_output_shapes
:?????????2!
model_12/graph_conv_38/MatMul_5?
(model_12/graph_conv_38/Reshape_5/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2*
(model_12/graph_conv_38/Reshape_5/shape/2?
&model_12/graph_conv_38/Reshape_5/shapePack)model_12/graph_conv_38/unstack_2:output:0)model_12/graph_conv_38/unstack_2:output:11model_12/graph_conv_38/Reshape_5/shape/2:output:0*
N*
T0*
_output_shapes
:2(
&model_12/graph_conv_38/Reshape_5/shape?
 model_12/graph_conv_38/Reshape_5Reshape)model_12/graph_conv_38/MatMul_5:product:0/model_12/graph_conv_38/Reshape_5/shape:output:0*
T0*4
_output_shapes"
 :??????????????????2"
 model_12/graph_conv_38/Reshape_5?
+model_12/graph_conv_38/add_5/ReadVariableOpReadVariableOp4model_12_graph_conv_38_add_1_readvariableop_resource*
_output_shapes
:*
dtype02-
+model_12/graph_conv_38/add_5/ReadVariableOp?
model_12/graph_conv_38/add_5AddV2)model_12/graph_conv_38/Reshape_5:output:03model_12/graph_conv_38/add_5/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :??????????????????2
model_12/graph_conv_38/add_5?
model_12/graph_conv_38/MatMul_6BatchMatMulV2!model_12/graph_conv_38/Cast_2:y:0!model_12/graph_conv_38/Cast_2:y:0*
T0*=
_output_shapes+
):'???????????????????????????2!
model_12/graph_conv_38/MatMul_6?
model_12/graph_conv_38/Shape_8Shape(model_12/graph_conv_38/MatMul_6:output:0*
T0*
_output_shapes
:2 
model_12/graph_conv_38/Shape_8?
model_12/graph_conv_38/add_6AddV2(model_12/graph_conv_38/MatMul_6:output:0!model_12/graph_conv_38/Cast_2:y:0*
T0*=
_output_shapes+
):'???????????????????????????2
model_12/graph_conv_38/add_6?
"model_12/graph_conv_38/Greater_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"model_12/graph_conv_38/Greater_3/y?
 model_12/graph_conv_38/Greater_3Greater model_12/graph_conv_38/add_6:z:0+model_12/graph_conv_38/Greater_3/y:output:0*
T0*=
_output_shapes+
):'???????????????????????????2"
 model_12/graph_conv_38/Greater_3?
model_12/graph_conv_38/Cast_3Cast$model_12/graph_conv_38/Greater_3:z:0*

DstT0*

SrcT0
*=
_output_shapes+
):'???????????????????????????2
model_12/graph_conv_38/Cast_3?
'model_12/graph_conv_38/transpose_3/permConst*
_output_shapes
:*
dtype0*!
valueB"          2)
'model_12/graph_conv_38/transpose_3/perm?
"model_12/graph_conv_38/transpose_3	Transpose!model_12/graph_conv_38/Cast_3:y:00model_12/graph_conv_38/transpose_3/perm:output:0*
T0*=
_output_shapes+
):'???????????????????????????2$
"model_12/graph_conv_38/transpose_3?
model_12/graph_conv_38/MatMul_7BatchMatMulV2&model_12/graph_conv_38/transpose_3:y:0 model_12/graph_conv_38/add_5:z:0*
T0*4
_output_shapes"
 :??????????????????2!
model_12/graph_conv_38/MatMul_7?
model_12/graph_conv_38/Shape_9Shape(model_12/graph_conv_38/MatMul_7:output:0*
T0*
_output_shapes
:2 
model_12/graph_conv_38/Shape_9?
.model_12/graph_conv_38/Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :20
.model_12/graph_conv_38/Sum_1/reduction_indices?
model_12/graph_conv_38/Sum_1Sum!model_12/graph_conv_38/Cast_3:y:07model_12/graph_conv_38/Sum_1/reduction_indices:output:0*
T0*4
_output_shapes"
 :??????????????????*
	keep_dims(2
model_12/graph_conv_38/Sum_1?
model_12/graph_conv_38/add_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32 
model_12/graph_conv_38/add_7/y?
model_12/graph_conv_38/add_7AddV2%model_12/graph_conv_38/Sum_1:output:0'model_12/graph_conv_38/add_7/y:output:0*
T0*4
_output_shapes"
 :??????????????????2
model_12/graph_conv_38/add_7?
 model_12/graph_conv_38/truediv_1RealDiv(model_12/graph_conv_38/MatMul_7:output:0 model_12/graph_conv_38/add_7:z:0*
T0*4
_output_shapes"
 :??????????????????2"
 model_12/graph_conv_38/truediv_1?
model_12/graph_conv_38/Relu_1Relu$model_12/graph_conv_38/truediv_1:z:0*
T0*4
_output_shapes"
 :??????????????????2
model_12/graph_conv_38/Relu_1?
8model_12/tf.__operators__.getitem_25/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2:
8model_12/tf.__operators__.getitem_25/strided_slice/stack?
:model_12/tf.__operators__.getitem_25/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2<
:model_12/tf.__operators__.getitem_25/strided_slice/stack_1?
:model_12/tf.__operators__.getitem_25/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2<
:model_12/tf.__operators__.getitem_25/strided_slice/stack_2?
2model_12/tf.__operators__.getitem_25/strided_sliceStridedSlice)model_12/graph_conv_38/Relu:activations:0Amodel_12/tf.__operators__.getitem_25/strided_slice/stack:output:0Cmodel_12/tf.__operators__.getitem_25/strided_slice/stack_1:output:0Cmodel_12/tf.__operators__.getitem_25/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask24
2model_12/tf.__operators__.getitem_25/strided_slice?
8model_12/tf.__operators__.getitem_24/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2:
8model_12/tf.__operators__.getitem_24/strided_slice/stack?
:model_12/tf.__operators__.getitem_24/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2<
:model_12/tf.__operators__.getitem_24/strided_slice/stack_1?
:model_12/tf.__operators__.getitem_24/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2<
:model_12/tf.__operators__.getitem_24/strided_slice/stack_2?
2model_12/tf.__operators__.getitem_24/strided_sliceStridedSlice+model_12/graph_conv_38/Relu_1:activations:0Amodel_12/tf.__operators__.getitem_24/strided_slice/stack:output:0Cmodel_12/tf.__operators__.getitem_24/strided_slice/stack_1:output:0Cmodel_12/tf.__operators__.getitem_24/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask24
2model_12/tf.__operators__.getitem_24/strided_slice?
+model_12/attention_12/MatMul/ReadVariableOpReadVariableOp4model_12_attention_12_matmul_readvariableop_resource*
_output_shapes

:*
dtype02-
+model_12/attention_12/MatMul/ReadVariableOp?
model_12/attention_12/MatMulMatMul;model_12/tf.__operators__.getitem_24/strided_slice:output:03model_12/attention_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_12/attention_12/MatMul?
,model_12/attention_12/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2.
,model_12/attention_12/Mean/reduction_indices?
model_12/attention_12/MeanMean&model_12/attention_12/MatMul:product:05model_12/attention_12/Mean/reduction_indices:output:0*
T0*
_output_shapes
:2
model_12/attention_12/Mean?
model_12/attention_12/TanhTanh#model_12/attention_12/Mean:output:0*
T0*
_output_shapes
:2
model_12/attention_12/Tanh?
#model_12/attention_12/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ????2%
#model_12/attention_12/Reshape/shape?
model_12/attention_12/ReshapeReshapemodel_12/attention_12/Tanh:y:0,model_12/attention_12/Reshape/shape:output:0*
T0*
_output_shapes

:2
model_12/attention_12/Reshape?
$model_12/attention_12/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2&
$model_12/attention_12/transpose/perm?
model_12/attention_12/transpose	Transpose&model_12/attention_12/Reshape:output:0-model_12/attention_12/transpose/perm:output:0*
T0*
_output_shapes

:2!
model_12/attention_12/transpose?
model_12/attention_12/MatMul_1MatMul;model_12/tf.__operators__.getitem_24/strided_slice:output:0#model_12/attention_12/transpose:y:0*
T0*'
_output_shapes
:?????????2 
model_12/attention_12/MatMul_1?
model_12/attention_12/SigmoidSigmoid(model_12/attention_12/MatMul_1:product:0*
T0*'
_output_shapes
:?????????2
model_12/attention_12/Sigmoid?
&model_12/attention_12/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2(
&model_12/attention_12/transpose_1/perm?
!model_12/attention_12/transpose_1	Transpose;model_12/tf.__operators__.getitem_24/strided_slice:output:0/model_12/attention_12/transpose_1/perm:output:0*
T0*'
_output_shapes
:?????????2#
!model_12/attention_12/transpose_1?
model_12/attention_12/MatMul_2MatMul%model_12/attention_12/transpose_1:y:0!model_12/attention_12/Sigmoid:y:0*
T0*
_output_shapes

:2 
model_12/attention_12/MatMul_2?
&model_12/attention_12/transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2(
&model_12/attention_12/transpose_2/perm?
!model_12/attention_12/transpose_2	Transpose(model_12/attention_12/MatMul_2:product:0/model_12/attention_12/transpose_2/perm:output:0*
T0*
_output_shapes

:2#
!model_12/attention_12/transpose_2?
-model_12/attention_12/MatMul_3/ReadVariableOpReadVariableOp4model_12_attention_12_matmul_readvariableop_resource*
_output_shapes

:*
dtype02/
-model_12/attention_12/MatMul_3/ReadVariableOp?
model_12/attention_12/MatMul_3MatMul;model_12/tf.__operators__.getitem_25/strided_slice:output:05model_12/attention_12/MatMul_3/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2 
model_12/attention_12/MatMul_3?
.model_12/attention_12/Mean_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 20
.model_12/attention_12/Mean_1/reduction_indices?
model_12/attention_12/Mean_1Mean(model_12/attention_12/MatMul_3:product:07model_12/attention_12/Mean_1/reduction_indices:output:0*
T0*
_output_shapes
:2
model_12/attention_12/Mean_1?
model_12/attention_12/Tanh_1Tanh%model_12/attention_12/Mean_1:output:0*
T0*
_output_shapes
:2
model_12/attention_12/Tanh_1?
%model_12/attention_12/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ????2'
%model_12/attention_12/Reshape_1/shape?
model_12/attention_12/Reshape_1Reshape model_12/attention_12/Tanh_1:y:0.model_12/attention_12/Reshape_1/shape:output:0*
T0*
_output_shapes

:2!
model_12/attention_12/Reshape_1?
&model_12/attention_12/transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2(
&model_12/attention_12/transpose_3/perm?
!model_12/attention_12/transpose_3	Transpose(model_12/attention_12/Reshape_1:output:0/model_12/attention_12/transpose_3/perm:output:0*
T0*
_output_shapes

:2#
!model_12/attention_12/transpose_3?
model_12/attention_12/MatMul_4MatMul;model_12/tf.__operators__.getitem_25/strided_slice:output:0%model_12/attention_12/transpose_3:y:0*
T0*'
_output_shapes
:?????????2 
model_12/attention_12/MatMul_4?
model_12/attention_12/Sigmoid_1Sigmoid(model_12/attention_12/MatMul_4:product:0*
T0*'
_output_shapes
:?????????2!
model_12/attention_12/Sigmoid_1?
&model_12/attention_12/transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       2(
&model_12/attention_12/transpose_4/perm?
!model_12/attention_12/transpose_4	Transpose;model_12/tf.__operators__.getitem_25/strided_slice:output:0/model_12/attention_12/transpose_4/perm:output:0*
T0*'
_output_shapes
:?????????2#
!model_12/attention_12/transpose_4?
model_12/attention_12/MatMul_5MatMul%model_12/attention_12/transpose_4:y:0#model_12/attention_12/Sigmoid_1:y:0*
T0*
_output_shapes

:2 
model_12/attention_12/MatMul_5?
&model_12/attention_12/transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       2(
&model_12/attention_12/transpose_5/perm?
!model_12/attention_12/transpose_5	Transpose(model_12/attention_12/MatMul_5:product:0/model_12/attention_12/transpose_5/perm:output:0*
T0*
_output_shapes

:2#
!model_12/attention_12/transpose_5?
%model_12/neural_tensor_layer_12/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2'
%model_12/neural_tensor_layer_12/Shape?
3model_12/neural_tensor_layer_12/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3model_12/neural_tensor_layer_12/strided_slice/stack?
5model_12/neural_tensor_layer_12/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5model_12/neural_tensor_layer_12/strided_slice/stack_1?
5model_12/neural_tensor_layer_12/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5model_12/neural_tensor_layer_12/strided_slice/stack_2?
-model_12/neural_tensor_layer_12/strided_sliceStridedSlice.model_12/neural_tensor_layer_12/Shape:output:0<model_12/neural_tensor_layer_12/strided_slice/stack:output:0>model_12/neural_tensor_layer_12/strided_slice/stack_1:output:0>model_12/neural_tensor_layer_12/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-model_12/neural_tensor_layer_12/strided_slice?
+model_12/neural_tensor_layer_12/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2-
+model_12/neural_tensor_layer_12/concat/axis?
&model_12/neural_tensor_layer_12/concatConcatV2%model_12/attention_12/transpose_2:y:0%model_12/attention_12/transpose_5:y:04model_12/neural_tensor_layer_12/concat/axis:output:0*
N*
T0*
_output_shapes

: 2(
&model_12/neural_tensor_layer_12/concat?
5model_12/neural_tensor_layer_12/MatMul/ReadVariableOpReadVariableOp>model_12_neural_tensor_layer_12_matmul_readvariableop_resource*
_output_shapes

: *
dtype027
5model_12/neural_tensor_layer_12/MatMul/ReadVariableOp?
&model_12/neural_tensor_layer_12/MatMulMatMul/model_12/neural_tensor_layer_12/concat:output:0=model_12/neural_tensor_layer_12/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2(
&model_12/neural_tensor_layer_12/MatMul?
.model_12/neural_tensor_layer_12/ReadVariableOpReadVariableOp7model_12_neural_tensor_layer_12_readvariableop_resource*"
_output_shapes
:*
dtype020
.model_12/neural_tensor_layer_12/ReadVariableOp?
5model_12/neural_tensor_layer_12/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5model_12/neural_tensor_layer_12/strided_slice_1/stack?
7model_12/neural_tensor_layer_12/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7model_12/neural_tensor_layer_12/strided_slice_1/stack_1?
7model_12/neural_tensor_layer_12/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7model_12/neural_tensor_layer_12/strided_slice_1/stack_2?
/model_12/neural_tensor_layer_12/strided_slice_1StridedSlice6model_12/neural_tensor_layer_12/ReadVariableOp:value:0>model_12/neural_tensor_layer_12/strided_slice_1/stack:output:0@model_12/neural_tensor_layer_12/strided_slice_1/stack_1:output:0@model_12/neural_tensor_layer_12/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask21
/model_12/neural_tensor_layer_12/strided_slice_1?
(model_12/neural_tensor_layer_12/MatMul_1MatMul%model_12/attention_12/transpose_2:y:08model_12/neural_tensor_layer_12/strided_slice_1:output:0*
T0*
_output_shapes

:2*
(model_12/neural_tensor_layer_12/MatMul_1?
#model_12/neural_tensor_layer_12/mulMul%model_12/attention_12/transpose_5:y:02model_12/neural_tensor_layer_12/MatMul_1:product:0*
T0*
_output_shapes

:2%
#model_12/neural_tensor_layer_12/mul?
2model_12/neural_tensor_layer_12/add/ReadVariableOpReadVariableOp;model_12_neural_tensor_layer_12_add_readvariableop_resource*
_output_shapes
:*
dtype024
2model_12/neural_tensor_layer_12/add/ReadVariableOp?
#model_12/neural_tensor_layer_12/addAddV2'model_12/neural_tensor_layer_12/mul:z:0:model_12/neural_tensor_layer_12/add/ReadVariableOp:value:0*
T0*
_output_shapes

:2%
#model_12/neural_tensor_layer_12/add?
5model_12/neural_tensor_layer_12/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :27
5model_12/neural_tensor_layer_12/Sum/reduction_indices?
#model_12/neural_tensor_layer_12/SumSum'model_12/neural_tensor_layer_12/add:z:0>model_12/neural_tensor_layer_12/Sum/reduction_indices:output:0*
T0*
_output_shapes
:2%
#model_12/neural_tensor_layer_12/Sum?
0model_12/neural_tensor_layer_12/ReadVariableOp_1ReadVariableOp7model_12_neural_tensor_layer_12_readvariableop_resource*"
_output_shapes
:*
dtype022
0model_12/neural_tensor_layer_12/ReadVariableOp_1?
5model_12/neural_tensor_layer_12/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:27
5model_12/neural_tensor_layer_12/strided_slice_2/stack?
7model_12/neural_tensor_layer_12/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7model_12/neural_tensor_layer_12/strided_slice_2/stack_1?
7model_12/neural_tensor_layer_12/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7model_12/neural_tensor_layer_12/strided_slice_2/stack_2?
/model_12/neural_tensor_layer_12/strided_slice_2StridedSlice8model_12/neural_tensor_layer_12/ReadVariableOp_1:value:0>model_12/neural_tensor_layer_12/strided_slice_2/stack:output:0@model_12/neural_tensor_layer_12/strided_slice_2/stack_1:output:0@model_12/neural_tensor_layer_12/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask21
/model_12/neural_tensor_layer_12/strided_slice_2?
(model_12/neural_tensor_layer_12/MatMul_2MatMul%model_12/attention_12/transpose_2:y:08model_12/neural_tensor_layer_12/strided_slice_2:output:0*
T0*
_output_shapes

:2*
(model_12/neural_tensor_layer_12/MatMul_2?
%model_12/neural_tensor_layer_12/mul_1Mul%model_12/attention_12/transpose_5:y:02model_12/neural_tensor_layer_12/MatMul_2:product:0*
T0*
_output_shapes

:2'
%model_12/neural_tensor_layer_12/mul_1?
4model_12/neural_tensor_layer_12/add_1/ReadVariableOpReadVariableOp;model_12_neural_tensor_layer_12_add_readvariableop_resource*
_output_shapes
:*
dtype026
4model_12/neural_tensor_layer_12/add_1/ReadVariableOp?
%model_12/neural_tensor_layer_12/add_1AddV2)model_12/neural_tensor_layer_12/mul_1:z:0<model_12/neural_tensor_layer_12/add_1/ReadVariableOp:value:0*
T0*
_output_shapes

:2'
%model_12/neural_tensor_layer_12/add_1?
7model_12/neural_tensor_layer_12/Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :29
7model_12/neural_tensor_layer_12/Sum_1/reduction_indices?
%model_12/neural_tensor_layer_12/Sum_1Sum)model_12/neural_tensor_layer_12/add_1:z:0@model_12/neural_tensor_layer_12/Sum_1/reduction_indices:output:0*
T0*
_output_shapes
:2'
%model_12/neural_tensor_layer_12/Sum_1?
0model_12/neural_tensor_layer_12/ReadVariableOp_2ReadVariableOp7model_12_neural_tensor_layer_12_readvariableop_resource*"
_output_shapes
:*
dtype022
0model_12/neural_tensor_layer_12/ReadVariableOp_2?
5model_12/neural_tensor_layer_12/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:27
5model_12/neural_tensor_layer_12/strided_slice_3/stack?
7model_12/neural_tensor_layer_12/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7model_12/neural_tensor_layer_12/strided_slice_3/stack_1?
7model_12/neural_tensor_layer_12/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7model_12/neural_tensor_layer_12/strided_slice_3/stack_2?
/model_12/neural_tensor_layer_12/strided_slice_3StridedSlice8model_12/neural_tensor_layer_12/ReadVariableOp_2:value:0>model_12/neural_tensor_layer_12/strided_slice_3/stack:output:0@model_12/neural_tensor_layer_12/strided_slice_3/stack_1:output:0@model_12/neural_tensor_layer_12/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask21
/model_12/neural_tensor_layer_12/strided_slice_3?
(model_12/neural_tensor_layer_12/MatMul_3MatMul%model_12/attention_12/transpose_2:y:08model_12/neural_tensor_layer_12/strided_slice_3:output:0*
T0*
_output_shapes

:2*
(model_12/neural_tensor_layer_12/MatMul_3?
%model_12/neural_tensor_layer_12/mul_2Mul%model_12/attention_12/transpose_5:y:02model_12/neural_tensor_layer_12/MatMul_3:product:0*
T0*
_output_shapes

:2'
%model_12/neural_tensor_layer_12/mul_2?
4model_12/neural_tensor_layer_12/add_2/ReadVariableOpReadVariableOp;model_12_neural_tensor_layer_12_add_readvariableop_resource*
_output_shapes
:*
dtype026
4model_12/neural_tensor_layer_12/add_2/ReadVariableOp?
%model_12/neural_tensor_layer_12/add_2AddV2)model_12/neural_tensor_layer_12/mul_2:z:0<model_12/neural_tensor_layer_12/add_2/ReadVariableOp:value:0*
T0*
_output_shapes

:2'
%model_12/neural_tensor_layer_12/add_2?
7model_12/neural_tensor_layer_12/Sum_2/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :29
7model_12/neural_tensor_layer_12/Sum_2/reduction_indices?
%model_12/neural_tensor_layer_12/Sum_2Sum)model_12/neural_tensor_layer_12/add_2:z:0@model_12/neural_tensor_layer_12/Sum_2/reduction_indices:output:0*
T0*
_output_shapes
:2'
%model_12/neural_tensor_layer_12/Sum_2?
0model_12/neural_tensor_layer_12/ReadVariableOp_3ReadVariableOp7model_12_neural_tensor_layer_12_readvariableop_resource*"
_output_shapes
:*
dtype022
0model_12/neural_tensor_layer_12/ReadVariableOp_3?
5model_12/neural_tensor_layer_12/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:27
5model_12/neural_tensor_layer_12/strided_slice_4/stack?
7model_12/neural_tensor_layer_12/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7model_12/neural_tensor_layer_12/strided_slice_4/stack_1?
7model_12/neural_tensor_layer_12/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7model_12/neural_tensor_layer_12/strided_slice_4/stack_2?
/model_12/neural_tensor_layer_12/strided_slice_4StridedSlice8model_12/neural_tensor_layer_12/ReadVariableOp_3:value:0>model_12/neural_tensor_layer_12/strided_slice_4/stack:output:0@model_12/neural_tensor_layer_12/strided_slice_4/stack_1:output:0@model_12/neural_tensor_layer_12/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask21
/model_12/neural_tensor_layer_12/strided_slice_4?
(model_12/neural_tensor_layer_12/MatMul_4MatMul%model_12/attention_12/transpose_2:y:08model_12/neural_tensor_layer_12/strided_slice_4:output:0*
T0*
_output_shapes

:2*
(model_12/neural_tensor_layer_12/MatMul_4?
%model_12/neural_tensor_layer_12/mul_3Mul%model_12/attention_12/transpose_5:y:02model_12/neural_tensor_layer_12/MatMul_4:product:0*
T0*
_output_shapes

:2'
%model_12/neural_tensor_layer_12/mul_3?
4model_12/neural_tensor_layer_12/add_3/ReadVariableOpReadVariableOp;model_12_neural_tensor_layer_12_add_readvariableop_resource*
_output_shapes
:*
dtype026
4model_12/neural_tensor_layer_12/add_3/ReadVariableOp?
%model_12/neural_tensor_layer_12/add_3AddV2)model_12/neural_tensor_layer_12/mul_3:z:0<model_12/neural_tensor_layer_12/add_3/ReadVariableOp:value:0*
T0*
_output_shapes

:2'
%model_12/neural_tensor_layer_12/add_3?
7model_12/neural_tensor_layer_12/Sum_3/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :29
7model_12/neural_tensor_layer_12/Sum_3/reduction_indices?
%model_12/neural_tensor_layer_12/Sum_3Sum)model_12/neural_tensor_layer_12/add_3:z:0@model_12/neural_tensor_layer_12/Sum_3/reduction_indices:output:0*
T0*
_output_shapes
:2'
%model_12/neural_tensor_layer_12/Sum_3?
0model_12/neural_tensor_layer_12/ReadVariableOp_4ReadVariableOp7model_12_neural_tensor_layer_12_readvariableop_resource*"
_output_shapes
:*
dtype022
0model_12/neural_tensor_layer_12/ReadVariableOp_4?
5model_12/neural_tensor_layer_12/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:27
5model_12/neural_tensor_layer_12/strided_slice_5/stack?
7model_12/neural_tensor_layer_12/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7model_12/neural_tensor_layer_12/strided_slice_5/stack_1?
7model_12/neural_tensor_layer_12/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7model_12/neural_tensor_layer_12/strided_slice_5/stack_2?
/model_12/neural_tensor_layer_12/strided_slice_5StridedSlice8model_12/neural_tensor_layer_12/ReadVariableOp_4:value:0>model_12/neural_tensor_layer_12/strided_slice_5/stack:output:0@model_12/neural_tensor_layer_12/strided_slice_5/stack_1:output:0@model_12/neural_tensor_layer_12/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask21
/model_12/neural_tensor_layer_12/strided_slice_5?
(model_12/neural_tensor_layer_12/MatMul_5MatMul%model_12/attention_12/transpose_2:y:08model_12/neural_tensor_layer_12/strided_slice_5:output:0*
T0*
_output_shapes

:2*
(model_12/neural_tensor_layer_12/MatMul_5?
%model_12/neural_tensor_layer_12/mul_4Mul%model_12/attention_12/transpose_5:y:02model_12/neural_tensor_layer_12/MatMul_5:product:0*
T0*
_output_shapes

:2'
%model_12/neural_tensor_layer_12/mul_4?
4model_12/neural_tensor_layer_12/add_4/ReadVariableOpReadVariableOp;model_12_neural_tensor_layer_12_add_readvariableop_resource*
_output_shapes
:*
dtype026
4model_12/neural_tensor_layer_12/add_4/ReadVariableOp?
%model_12/neural_tensor_layer_12/add_4AddV2)model_12/neural_tensor_layer_12/mul_4:z:0<model_12/neural_tensor_layer_12/add_4/ReadVariableOp:value:0*
T0*
_output_shapes

:2'
%model_12/neural_tensor_layer_12/add_4?
7model_12/neural_tensor_layer_12/Sum_4/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :29
7model_12/neural_tensor_layer_12/Sum_4/reduction_indices?
%model_12/neural_tensor_layer_12/Sum_4Sum)model_12/neural_tensor_layer_12/add_4:z:0@model_12/neural_tensor_layer_12/Sum_4/reduction_indices:output:0*
T0*
_output_shapes
:2'
%model_12/neural_tensor_layer_12/Sum_4?
0model_12/neural_tensor_layer_12/ReadVariableOp_5ReadVariableOp7model_12_neural_tensor_layer_12_readvariableop_resource*"
_output_shapes
:*
dtype022
0model_12/neural_tensor_layer_12/ReadVariableOp_5?
5model_12/neural_tensor_layer_12/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB:27
5model_12/neural_tensor_layer_12/strided_slice_6/stack?
7model_12/neural_tensor_layer_12/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7model_12/neural_tensor_layer_12/strided_slice_6/stack_1?
7model_12/neural_tensor_layer_12/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7model_12/neural_tensor_layer_12/strided_slice_6/stack_2?
/model_12/neural_tensor_layer_12/strided_slice_6StridedSlice8model_12/neural_tensor_layer_12/ReadVariableOp_5:value:0>model_12/neural_tensor_layer_12/strided_slice_6/stack:output:0@model_12/neural_tensor_layer_12/strided_slice_6/stack_1:output:0@model_12/neural_tensor_layer_12/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask21
/model_12/neural_tensor_layer_12/strided_slice_6?
(model_12/neural_tensor_layer_12/MatMul_6MatMul%model_12/attention_12/transpose_2:y:08model_12/neural_tensor_layer_12/strided_slice_6:output:0*
T0*
_output_shapes

:2*
(model_12/neural_tensor_layer_12/MatMul_6?
%model_12/neural_tensor_layer_12/mul_5Mul%model_12/attention_12/transpose_5:y:02model_12/neural_tensor_layer_12/MatMul_6:product:0*
T0*
_output_shapes

:2'
%model_12/neural_tensor_layer_12/mul_5?
4model_12/neural_tensor_layer_12/add_5/ReadVariableOpReadVariableOp;model_12_neural_tensor_layer_12_add_readvariableop_resource*
_output_shapes
:*
dtype026
4model_12/neural_tensor_layer_12/add_5/ReadVariableOp?
%model_12/neural_tensor_layer_12/add_5AddV2)model_12/neural_tensor_layer_12/mul_5:z:0<model_12/neural_tensor_layer_12/add_5/ReadVariableOp:value:0*
T0*
_output_shapes

:2'
%model_12/neural_tensor_layer_12/add_5?
7model_12/neural_tensor_layer_12/Sum_5/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :29
7model_12/neural_tensor_layer_12/Sum_5/reduction_indices?
%model_12/neural_tensor_layer_12/Sum_5Sum)model_12/neural_tensor_layer_12/add_5:z:0@model_12/neural_tensor_layer_12/Sum_5/reduction_indices:output:0*
T0*
_output_shapes
:2'
%model_12/neural_tensor_layer_12/Sum_5?
0model_12/neural_tensor_layer_12/ReadVariableOp_6ReadVariableOp7model_12_neural_tensor_layer_12_readvariableop_resource*"
_output_shapes
:*
dtype022
0model_12/neural_tensor_layer_12/ReadVariableOp_6?
5model_12/neural_tensor_layer_12/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB:27
5model_12/neural_tensor_layer_12/strided_slice_7/stack?
7model_12/neural_tensor_layer_12/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7model_12/neural_tensor_layer_12/strided_slice_7/stack_1?
7model_12/neural_tensor_layer_12/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7model_12/neural_tensor_layer_12/strided_slice_7/stack_2?
/model_12/neural_tensor_layer_12/strided_slice_7StridedSlice8model_12/neural_tensor_layer_12/ReadVariableOp_6:value:0>model_12/neural_tensor_layer_12/strided_slice_7/stack:output:0@model_12/neural_tensor_layer_12/strided_slice_7/stack_1:output:0@model_12/neural_tensor_layer_12/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask21
/model_12/neural_tensor_layer_12/strided_slice_7?
(model_12/neural_tensor_layer_12/MatMul_7MatMul%model_12/attention_12/transpose_2:y:08model_12/neural_tensor_layer_12/strided_slice_7:output:0*
T0*
_output_shapes

:2*
(model_12/neural_tensor_layer_12/MatMul_7?
%model_12/neural_tensor_layer_12/mul_6Mul%model_12/attention_12/transpose_5:y:02model_12/neural_tensor_layer_12/MatMul_7:product:0*
T0*
_output_shapes

:2'
%model_12/neural_tensor_layer_12/mul_6?
4model_12/neural_tensor_layer_12/add_6/ReadVariableOpReadVariableOp;model_12_neural_tensor_layer_12_add_readvariableop_resource*
_output_shapes
:*
dtype026
4model_12/neural_tensor_layer_12/add_6/ReadVariableOp?
%model_12/neural_tensor_layer_12/add_6AddV2)model_12/neural_tensor_layer_12/mul_6:z:0<model_12/neural_tensor_layer_12/add_6/ReadVariableOp:value:0*
T0*
_output_shapes

:2'
%model_12/neural_tensor_layer_12/add_6?
7model_12/neural_tensor_layer_12/Sum_6/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :29
7model_12/neural_tensor_layer_12/Sum_6/reduction_indices?
%model_12/neural_tensor_layer_12/Sum_6Sum)model_12/neural_tensor_layer_12/add_6:z:0@model_12/neural_tensor_layer_12/Sum_6/reduction_indices:output:0*
T0*
_output_shapes
:2'
%model_12/neural_tensor_layer_12/Sum_6?
0model_12/neural_tensor_layer_12/ReadVariableOp_7ReadVariableOp7model_12_neural_tensor_layer_12_readvariableop_resource*"
_output_shapes
:*
dtype022
0model_12/neural_tensor_layer_12/ReadVariableOp_7?
5model_12/neural_tensor_layer_12/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB:27
5model_12/neural_tensor_layer_12/strided_slice_8/stack?
7model_12/neural_tensor_layer_12/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7model_12/neural_tensor_layer_12/strided_slice_8/stack_1?
7model_12/neural_tensor_layer_12/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7model_12/neural_tensor_layer_12/strided_slice_8/stack_2?
/model_12/neural_tensor_layer_12/strided_slice_8StridedSlice8model_12/neural_tensor_layer_12/ReadVariableOp_7:value:0>model_12/neural_tensor_layer_12/strided_slice_8/stack:output:0@model_12/neural_tensor_layer_12/strided_slice_8/stack_1:output:0@model_12/neural_tensor_layer_12/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask21
/model_12/neural_tensor_layer_12/strided_slice_8?
(model_12/neural_tensor_layer_12/MatMul_8MatMul%model_12/attention_12/transpose_2:y:08model_12/neural_tensor_layer_12/strided_slice_8:output:0*
T0*
_output_shapes

:2*
(model_12/neural_tensor_layer_12/MatMul_8?
%model_12/neural_tensor_layer_12/mul_7Mul%model_12/attention_12/transpose_5:y:02model_12/neural_tensor_layer_12/MatMul_8:product:0*
T0*
_output_shapes

:2'
%model_12/neural_tensor_layer_12/mul_7?
4model_12/neural_tensor_layer_12/add_7/ReadVariableOpReadVariableOp;model_12_neural_tensor_layer_12_add_readvariableop_resource*
_output_shapes
:*
dtype026
4model_12/neural_tensor_layer_12/add_7/ReadVariableOp?
%model_12/neural_tensor_layer_12/add_7AddV2)model_12/neural_tensor_layer_12/mul_7:z:0<model_12/neural_tensor_layer_12/add_7/ReadVariableOp:value:0*
T0*
_output_shapes

:2'
%model_12/neural_tensor_layer_12/add_7?
7model_12/neural_tensor_layer_12/Sum_7/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :29
7model_12/neural_tensor_layer_12/Sum_7/reduction_indices?
%model_12/neural_tensor_layer_12/Sum_7Sum)model_12/neural_tensor_layer_12/add_7:z:0@model_12/neural_tensor_layer_12/Sum_7/reduction_indices:output:0*
T0*
_output_shapes
:2'
%model_12/neural_tensor_layer_12/Sum_7?
0model_12/neural_tensor_layer_12/ReadVariableOp_8ReadVariableOp7model_12_neural_tensor_layer_12_readvariableop_resource*"
_output_shapes
:*
dtype022
0model_12/neural_tensor_layer_12/ReadVariableOp_8?
5model_12/neural_tensor_layer_12/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB:27
5model_12/neural_tensor_layer_12/strided_slice_9/stack?
7model_12/neural_tensor_layer_12/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:	29
7model_12/neural_tensor_layer_12/strided_slice_9/stack_1?
7model_12/neural_tensor_layer_12/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7model_12/neural_tensor_layer_12/strided_slice_9/stack_2?
/model_12/neural_tensor_layer_12/strided_slice_9StridedSlice8model_12/neural_tensor_layer_12/ReadVariableOp_8:value:0>model_12/neural_tensor_layer_12/strided_slice_9/stack:output:0@model_12/neural_tensor_layer_12/strided_slice_9/stack_1:output:0@model_12/neural_tensor_layer_12/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask21
/model_12/neural_tensor_layer_12/strided_slice_9?
(model_12/neural_tensor_layer_12/MatMul_9MatMul%model_12/attention_12/transpose_2:y:08model_12/neural_tensor_layer_12/strided_slice_9:output:0*
T0*
_output_shapes

:2*
(model_12/neural_tensor_layer_12/MatMul_9?
%model_12/neural_tensor_layer_12/mul_8Mul%model_12/attention_12/transpose_5:y:02model_12/neural_tensor_layer_12/MatMul_9:product:0*
T0*
_output_shapes

:2'
%model_12/neural_tensor_layer_12/mul_8?
4model_12/neural_tensor_layer_12/add_8/ReadVariableOpReadVariableOp;model_12_neural_tensor_layer_12_add_readvariableop_resource*
_output_shapes
:*
dtype026
4model_12/neural_tensor_layer_12/add_8/ReadVariableOp?
%model_12/neural_tensor_layer_12/add_8AddV2)model_12/neural_tensor_layer_12/mul_8:z:0<model_12/neural_tensor_layer_12/add_8/ReadVariableOp:value:0*
T0*
_output_shapes

:2'
%model_12/neural_tensor_layer_12/add_8?
7model_12/neural_tensor_layer_12/Sum_8/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :29
7model_12/neural_tensor_layer_12/Sum_8/reduction_indices?
%model_12/neural_tensor_layer_12/Sum_8Sum)model_12/neural_tensor_layer_12/add_8:z:0@model_12/neural_tensor_layer_12/Sum_8/reduction_indices:output:0*
T0*
_output_shapes
:2'
%model_12/neural_tensor_layer_12/Sum_8?
0model_12/neural_tensor_layer_12/ReadVariableOp_9ReadVariableOp7model_12_neural_tensor_layer_12_readvariableop_resource*"
_output_shapes
:*
dtype022
0model_12/neural_tensor_layer_12/ReadVariableOp_9?
6model_12/neural_tensor_layer_12/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB:	28
6model_12/neural_tensor_layer_12/strided_slice_10/stack?
8model_12/neural_tensor_layer_12/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
2:
8model_12/neural_tensor_layer_12/strided_slice_10/stack_1?
8model_12/neural_tensor_layer_12/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2:
8model_12/neural_tensor_layer_12/strided_slice_10/stack_2?
0model_12/neural_tensor_layer_12/strided_slice_10StridedSlice8model_12/neural_tensor_layer_12/ReadVariableOp_9:value:0?model_12/neural_tensor_layer_12/strided_slice_10/stack:output:0Amodel_12/neural_tensor_layer_12/strided_slice_10/stack_1:output:0Amodel_12/neural_tensor_layer_12/strided_slice_10/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask22
0model_12/neural_tensor_layer_12/strided_slice_10?
)model_12/neural_tensor_layer_12/MatMul_10MatMul%model_12/attention_12/transpose_2:y:09model_12/neural_tensor_layer_12/strided_slice_10:output:0*
T0*
_output_shapes

:2+
)model_12/neural_tensor_layer_12/MatMul_10?
%model_12/neural_tensor_layer_12/mul_9Mul%model_12/attention_12/transpose_5:y:03model_12/neural_tensor_layer_12/MatMul_10:product:0*
T0*
_output_shapes

:2'
%model_12/neural_tensor_layer_12/mul_9?
4model_12/neural_tensor_layer_12/add_9/ReadVariableOpReadVariableOp;model_12_neural_tensor_layer_12_add_readvariableop_resource*
_output_shapes
:*
dtype026
4model_12/neural_tensor_layer_12/add_9/ReadVariableOp?
%model_12/neural_tensor_layer_12/add_9AddV2)model_12/neural_tensor_layer_12/mul_9:z:0<model_12/neural_tensor_layer_12/add_9/ReadVariableOp:value:0*
T0*
_output_shapes

:2'
%model_12/neural_tensor_layer_12/add_9?
7model_12/neural_tensor_layer_12/Sum_9/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :29
7model_12/neural_tensor_layer_12/Sum_9/reduction_indices?
%model_12/neural_tensor_layer_12/Sum_9Sum)model_12/neural_tensor_layer_12/add_9:z:0@model_12/neural_tensor_layer_12/Sum_9/reduction_indices:output:0*
T0*
_output_shapes
:2'
%model_12/neural_tensor_layer_12/Sum_9?
1model_12/neural_tensor_layer_12/ReadVariableOp_10ReadVariableOp7model_12_neural_tensor_layer_12_readvariableop_resource*"
_output_shapes
:*
dtype023
1model_12/neural_tensor_layer_12/ReadVariableOp_10?
6model_12/neural_tensor_layer_12/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:
28
6model_12/neural_tensor_layer_12/strided_slice_11/stack?
8model_12/neural_tensor_layer_12/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2:
8model_12/neural_tensor_layer_12/strided_slice_11/stack_1?
8model_12/neural_tensor_layer_12/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2:
8model_12/neural_tensor_layer_12/strided_slice_11/stack_2?
0model_12/neural_tensor_layer_12/strided_slice_11StridedSlice9model_12/neural_tensor_layer_12/ReadVariableOp_10:value:0?model_12/neural_tensor_layer_12/strided_slice_11/stack:output:0Amodel_12/neural_tensor_layer_12/strided_slice_11/stack_1:output:0Amodel_12/neural_tensor_layer_12/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask22
0model_12/neural_tensor_layer_12/strided_slice_11?
)model_12/neural_tensor_layer_12/MatMul_11MatMul%model_12/attention_12/transpose_2:y:09model_12/neural_tensor_layer_12/strided_slice_11:output:0*
T0*
_output_shapes

:2+
)model_12/neural_tensor_layer_12/MatMul_11?
&model_12/neural_tensor_layer_12/mul_10Mul%model_12/attention_12/transpose_5:y:03model_12/neural_tensor_layer_12/MatMul_11:product:0*
T0*
_output_shapes

:2(
&model_12/neural_tensor_layer_12/mul_10?
5model_12/neural_tensor_layer_12/add_10/ReadVariableOpReadVariableOp;model_12_neural_tensor_layer_12_add_readvariableop_resource*
_output_shapes
:*
dtype027
5model_12/neural_tensor_layer_12/add_10/ReadVariableOp?
&model_12/neural_tensor_layer_12/add_10AddV2*model_12/neural_tensor_layer_12/mul_10:z:0=model_12/neural_tensor_layer_12/add_10/ReadVariableOp:value:0*
T0*
_output_shapes

:2(
&model_12/neural_tensor_layer_12/add_10?
8model_12/neural_tensor_layer_12/Sum_10/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2:
8model_12/neural_tensor_layer_12/Sum_10/reduction_indices?
&model_12/neural_tensor_layer_12/Sum_10Sum*model_12/neural_tensor_layer_12/add_10:z:0Amodel_12/neural_tensor_layer_12/Sum_10/reduction_indices:output:0*
T0*
_output_shapes
:2(
&model_12/neural_tensor_layer_12/Sum_10?
1model_12/neural_tensor_layer_12/ReadVariableOp_11ReadVariableOp7model_12_neural_tensor_layer_12_readvariableop_resource*"
_output_shapes
:*
dtype023
1model_12/neural_tensor_layer_12/ReadVariableOp_11?
6model_12/neural_tensor_layer_12/strided_slice_12/stackConst*
_output_shapes
:*
dtype0*
valueB:28
6model_12/neural_tensor_layer_12/strided_slice_12/stack?
8model_12/neural_tensor_layer_12/strided_slice_12/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2:
8model_12/neural_tensor_layer_12/strided_slice_12/stack_1?
8model_12/neural_tensor_layer_12/strided_slice_12/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2:
8model_12/neural_tensor_layer_12/strided_slice_12/stack_2?
0model_12/neural_tensor_layer_12/strided_slice_12StridedSlice9model_12/neural_tensor_layer_12/ReadVariableOp_11:value:0?model_12/neural_tensor_layer_12/strided_slice_12/stack:output:0Amodel_12/neural_tensor_layer_12/strided_slice_12/stack_1:output:0Amodel_12/neural_tensor_layer_12/strided_slice_12/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask22
0model_12/neural_tensor_layer_12/strided_slice_12?
)model_12/neural_tensor_layer_12/MatMul_12MatMul%model_12/attention_12/transpose_2:y:09model_12/neural_tensor_layer_12/strided_slice_12:output:0*
T0*
_output_shapes

:2+
)model_12/neural_tensor_layer_12/MatMul_12?
&model_12/neural_tensor_layer_12/mul_11Mul%model_12/attention_12/transpose_5:y:03model_12/neural_tensor_layer_12/MatMul_12:product:0*
T0*
_output_shapes

:2(
&model_12/neural_tensor_layer_12/mul_11?
5model_12/neural_tensor_layer_12/add_11/ReadVariableOpReadVariableOp;model_12_neural_tensor_layer_12_add_readvariableop_resource*
_output_shapes
:*
dtype027
5model_12/neural_tensor_layer_12/add_11/ReadVariableOp?
&model_12/neural_tensor_layer_12/add_11AddV2*model_12/neural_tensor_layer_12/mul_11:z:0=model_12/neural_tensor_layer_12/add_11/ReadVariableOp:value:0*
T0*
_output_shapes

:2(
&model_12/neural_tensor_layer_12/add_11?
8model_12/neural_tensor_layer_12/Sum_11/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2:
8model_12/neural_tensor_layer_12/Sum_11/reduction_indices?
&model_12/neural_tensor_layer_12/Sum_11Sum*model_12/neural_tensor_layer_12/add_11:z:0Amodel_12/neural_tensor_layer_12/Sum_11/reduction_indices:output:0*
T0*
_output_shapes
:2(
&model_12/neural_tensor_layer_12/Sum_11?
1model_12/neural_tensor_layer_12/ReadVariableOp_12ReadVariableOp7model_12_neural_tensor_layer_12_readvariableop_resource*"
_output_shapes
:*
dtype023
1model_12/neural_tensor_layer_12/ReadVariableOp_12?
6model_12/neural_tensor_layer_12/strided_slice_13/stackConst*
_output_shapes
:*
dtype0*
valueB:28
6model_12/neural_tensor_layer_12/strided_slice_13/stack?
8model_12/neural_tensor_layer_12/strided_slice_13/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2:
8model_12/neural_tensor_layer_12/strided_slice_13/stack_1?
8model_12/neural_tensor_layer_12/strided_slice_13/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2:
8model_12/neural_tensor_layer_12/strided_slice_13/stack_2?
0model_12/neural_tensor_layer_12/strided_slice_13StridedSlice9model_12/neural_tensor_layer_12/ReadVariableOp_12:value:0?model_12/neural_tensor_layer_12/strided_slice_13/stack:output:0Amodel_12/neural_tensor_layer_12/strided_slice_13/stack_1:output:0Amodel_12/neural_tensor_layer_12/strided_slice_13/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask22
0model_12/neural_tensor_layer_12/strided_slice_13?
)model_12/neural_tensor_layer_12/MatMul_13MatMul%model_12/attention_12/transpose_2:y:09model_12/neural_tensor_layer_12/strided_slice_13:output:0*
T0*
_output_shapes

:2+
)model_12/neural_tensor_layer_12/MatMul_13?
&model_12/neural_tensor_layer_12/mul_12Mul%model_12/attention_12/transpose_5:y:03model_12/neural_tensor_layer_12/MatMul_13:product:0*
T0*
_output_shapes

:2(
&model_12/neural_tensor_layer_12/mul_12?
5model_12/neural_tensor_layer_12/add_12/ReadVariableOpReadVariableOp;model_12_neural_tensor_layer_12_add_readvariableop_resource*
_output_shapes
:*
dtype027
5model_12/neural_tensor_layer_12/add_12/ReadVariableOp?
&model_12/neural_tensor_layer_12/add_12AddV2*model_12/neural_tensor_layer_12/mul_12:z:0=model_12/neural_tensor_layer_12/add_12/ReadVariableOp:value:0*
T0*
_output_shapes

:2(
&model_12/neural_tensor_layer_12/add_12?
8model_12/neural_tensor_layer_12/Sum_12/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2:
8model_12/neural_tensor_layer_12/Sum_12/reduction_indices?
&model_12/neural_tensor_layer_12/Sum_12Sum*model_12/neural_tensor_layer_12/add_12:z:0Amodel_12/neural_tensor_layer_12/Sum_12/reduction_indices:output:0*
T0*
_output_shapes
:2(
&model_12/neural_tensor_layer_12/Sum_12?
1model_12/neural_tensor_layer_12/ReadVariableOp_13ReadVariableOp7model_12_neural_tensor_layer_12_readvariableop_resource*"
_output_shapes
:*
dtype023
1model_12/neural_tensor_layer_12/ReadVariableOp_13?
6model_12/neural_tensor_layer_12/strided_slice_14/stackConst*
_output_shapes
:*
dtype0*
valueB:28
6model_12/neural_tensor_layer_12/strided_slice_14/stack?
8model_12/neural_tensor_layer_12/strided_slice_14/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2:
8model_12/neural_tensor_layer_12/strided_slice_14/stack_1?
8model_12/neural_tensor_layer_12/strided_slice_14/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2:
8model_12/neural_tensor_layer_12/strided_slice_14/stack_2?
0model_12/neural_tensor_layer_12/strided_slice_14StridedSlice9model_12/neural_tensor_layer_12/ReadVariableOp_13:value:0?model_12/neural_tensor_layer_12/strided_slice_14/stack:output:0Amodel_12/neural_tensor_layer_12/strided_slice_14/stack_1:output:0Amodel_12/neural_tensor_layer_12/strided_slice_14/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask22
0model_12/neural_tensor_layer_12/strided_slice_14?
)model_12/neural_tensor_layer_12/MatMul_14MatMul%model_12/attention_12/transpose_2:y:09model_12/neural_tensor_layer_12/strided_slice_14:output:0*
T0*
_output_shapes

:2+
)model_12/neural_tensor_layer_12/MatMul_14?
&model_12/neural_tensor_layer_12/mul_13Mul%model_12/attention_12/transpose_5:y:03model_12/neural_tensor_layer_12/MatMul_14:product:0*
T0*
_output_shapes

:2(
&model_12/neural_tensor_layer_12/mul_13?
5model_12/neural_tensor_layer_12/add_13/ReadVariableOpReadVariableOp;model_12_neural_tensor_layer_12_add_readvariableop_resource*
_output_shapes
:*
dtype027
5model_12/neural_tensor_layer_12/add_13/ReadVariableOp?
&model_12/neural_tensor_layer_12/add_13AddV2*model_12/neural_tensor_layer_12/mul_13:z:0=model_12/neural_tensor_layer_12/add_13/ReadVariableOp:value:0*
T0*
_output_shapes

:2(
&model_12/neural_tensor_layer_12/add_13?
8model_12/neural_tensor_layer_12/Sum_13/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2:
8model_12/neural_tensor_layer_12/Sum_13/reduction_indices?
&model_12/neural_tensor_layer_12/Sum_13Sum*model_12/neural_tensor_layer_12/add_13:z:0Amodel_12/neural_tensor_layer_12/Sum_13/reduction_indices:output:0*
T0*
_output_shapes
:2(
&model_12/neural_tensor_layer_12/Sum_13?
1model_12/neural_tensor_layer_12/ReadVariableOp_14ReadVariableOp7model_12_neural_tensor_layer_12_readvariableop_resource*"
_output_shapes
:*
dtype023
1model_12/neural_tensor_layer_12/ReadVariableOp_14?
6model_12/neural_tensor_layer_12/strided_slice_15/stackConst*
_output_shapes
:*
dtype0*
valueB:28
6model_12/neural_tensor_layer_12/strided_slice_15/stack?
8model_12/neural_tensor_layer_12/strided_slice_15/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2:
8model_12/neural_tensor_layer_12/strided_slice_15/stack_1?
8model_12/neural_tensor_layer_12/strided_slice_15/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2:
8model_12/neural_tensor_layer_12/strided_slice_15/stack_2?
0model_12/neural_tensor_layer_12/strided_slice_15StridedSlice9model_12/neural_tensor_layer_12/ReadVariableOp_14:value:0?model_12/neural_tensor_layer_12/strided_slice_15/stack:output:0Amodel_12/neural_tensor_layer_12/strided_slice_15/stack_1:output:0Amodel_12/neural_tensor_layer_12/strided_slice_15/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask22
0model_12/neural_tensor_layer_12/strided_slice_15?
)model_12/neural_tensor_layer_12/MatMul_15MatMul%model_12/attention_12/transpose_2:y:09model_12/neural_tensor_layer_12/strided_slice_15:output:0*
T0*
_output_shapes

:2+
)model_12/neural_tensor_layer_12/MatMul_15?
&model_12/neural_tensor_layer_12/mul_14Mul%model_12/attention_12/transpose_5:y:03model_12/neural_tensor_layer_12/MatMul_15:product:0*
T0*
_output_shapes

:2(
&model_12/neural_tensor_layer_12/mul_14?
5model_12/neural_tensor_layer_12/add_14/ReadVariableOpReadVariableOp;model_12_neural_tensor_layer_12_add_readvariableop_resource*
_output_shapes
:*
dtype027
5model_12/neural_tensor_layer_12/add_14/ReadVariableOp?
&model_12/neural_tensor_layer_12/add_14AddV2*model_12/neural_tensor_layer_12/mul_14:z:0=model_12/neural_tensor_layer_12/add_14/ReadVariableOp:value:0*
T0*
_output_shapes

:2(
&model_12/neural_tensor_layer_12/add_14?
8model_12/neural_tensor_layer_12/Sum_14/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2:
8model_12/neural_tensor_layer_12/Sum_14/reduction_indices?
&model_12/neural_tensor_layer_12/Sum_14Sum*model_12/neural_tensor_layer_12/add_14:z:0Amodel_12/neural_tensor_layer_12/Sum_14/reduction_indices:output:0*
T0*
_output_shapes
:2(
&model_12/neural_tensor_layer_12/Sum_14?
1model_12/neural_tensor_layer_12/ReadVariableOp_15ReadVariableOp7model_12_neural_tensor_layer_12_readvariableop_resource*"
_output_shapes
:*
dtype023
1model_12/neural_tensor_layer_12/ReadVariableOp_15?
6model_12/neural_tensor_layer_12/strided_slice_16/stackConst*
_output_shapes
:*
dtype0*
valueB:28
6model_12/neural_tensor_layer_12/strided_slice_16/stack?
8model_12/neural_tensor_layer_12/strided_slice_16/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2:
8model_12/neural_tensor_layer_12/strided_slice_16/stack_1?
8model_12/neural_tensor_layer_12/strided_slice_16/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2:
8model_12/neural_tensor_layer_12/strided_slice_16/stack_2?
0model_12/neural_tensor_layer_12/strided_slice_16StridedSlice9model_12/neural_tensor_layer_12/ReadVariableOp_15:value:0?model_12/neural_tensor_layer_12/strided_slice_16/stack:output:0Amodel_12/neural_tensor_layer_12/strided_slice_16/stack_1:output:0Amodel_12/neural_tensor_layer_12/strided_slice_16/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask22
0model_12/neural_tensor_layer_12/strided_slice_16?
)model_12/neural_tensor_layer_12/MatMul_16MatMul%model_12/attention_12/transpose_2:y:09model_12/neural_tensor_layer_12/strided_slice_16:output:0*
T0*
_output_shapes

:2+
)model_12/neural_tensor_layer_12/MatMul_16?
&model_12/neural_tensor_layer_12/mul_15Mul%model_12/attention_12/transpose_5:y:03model_12/neural_tensor_layer_12/MatMul_16:product:0*
T0*
_output_shapes

:2(
&model_12/neural_tensor_layer_12/mul_15?
5model_12/neural_tensor_layer_12/add_15/ReadVariableOpReadVariableOp;model_12_neural_tensor_layer_12_add_readvariableop_resource*
_output_shapes
:*
dtype027
5model_12/neural_tensor_layer_12/add_15/ReadVariableOp?
&model_12/neural_tensor_layer_12/add_15AddV2*model_12/neural_tensor_layer_12/mul_15:z:0=model_12/neural_tensor_layer_12/add_15/ReadVariableOp:value:0*
T0*
_output_shapes

:2(
&model_12/neural_tensor_layer_12/add_15?
8model_12/neural_tensor_layer_12/Sum_15/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2:
8model_12/neural_tensor_layer_12/Sum_15/reduction_indices?
&model_12/neural_tensor_layer_12/Sum_15Sum*model_12/neural_tensor_layer_12/add_15:z:0Amodel_12/neural_tensor_layer_12/Sum_15/reduction_indices:output:0*
T0*
_output_shapes
:2(
&model_12/neural_tensor_layer_12/Sum_15?
-model_12/neural_tensor_layer_12/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2/
-model_12/neural_tensor_layer_12/concat_1/axis?
(model_12/neural_tensor_layer_12/concat_1ConcatV2,model_12/neural_tensor_layer_12/Sum:output:0.model_12/neural_tensor_layer_12/Sum_1:output:0.model_12/neural_tensor_layer_12/Sum_2:output:0.model_12/neural_tensor_layer_12/Sum_3:output:0.model_12/neural_tensor_layer_12/Sum_4:output:0.model_12/neural_tensor_layer_12/Sum_5:output:0.model_12/neural_tensor_layer_12/Sum_6:output:0.model_12/neural_tensor_layer_12/Sum_7:output:0.model_12/neural_tensor_layer_12/Sum_8:output:0.model_12/neural_tensor_layer_12/Sum_9:output:0/model_12/neural_tensor_layer_12/Sum_10:output:0/model_12/neural_tensor_layer_12/Sum_11:output:0/model_12/neural_tensor_layer_12/Sum_12:output:0/model_12/neural_tensor_layer_12/Sum_13:output:0/model_12/neural_tensor_layer_12/Sum_14:output:0/model_12/neural_tensor_layer_12/Sum_15:output:06model_12/neural_tensor_layer_12/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2*
(model_12/neural_tensor_layer_12/concat_1?
/model_12/neural_tensor_layer_12/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :21
/model_12/neural_tensor_layer_12/Reshape/shape/1?
-model_12/neural_tensor_layer_12/Reshape/shapePack6model_12/neural_tensor_layer_12/strided_slice:output:08model_12/neural_tensor_layer_12/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2/
-model_12/neural_tensor_layer_12/Reshape/shape?
'model_12/neural_tensor_layer_12/ReshapeReshape1model_12/neural_tensor_layer_12/concat_1:output:06model_12/neural_tensor_layer_12/Reshape/shape:output:0*
T0*
_output_shapes

:2)
'model_12/neural_tensor_layer_12/Reshape?
&model_12/neural_tensor_layer_12/add_16AddV20model_12/neural_tensor_layer_12/Reshape:output:00model_12/neural_tensor_layer_12/MatMul:product:0*
T0*
_output_shapes

:2(
&model_12/neural_tensor_layer_12/add_16?
$model_12/neural_tensor_layer_12/TanhTanh*model_12/neural_tensor_layer_12/add_16:z:0*
T0*
_output_shapes

:2&
$model_12/neural_tensor_layer_12/Tanh?
'model_12/dense_48/MatMul/ReadVariableOpReadVariableOp0model_12_dense_48_matmul_readvariableop_resource*
_output_shapes

:*
dtype02)
'model_12/dense_48/MatMul/ReadVariableOp?
model_12/dense_48/MatMulMatMul(model_12/neural_tensor_layer_12/Tanh:y:0/model_12/dense_48/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
model_12/dense_48/MatMul?
(model_12/dense_48/BiasAdd/ReadVariableOpReadVariableOp1model_12_dense_48_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(model_12/dense_48/BiasAdd/ReadVariableOp?
model_12/dense_48/BiasAddBiasAdd"model_12/dense_48/MatMul:product:00model_12/dense_48/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
model_12/dense_48/BiasAdd?
model_12/dense_48/ReluRelu"model_12/dense_48/BiasAdd:output:0*
T0*
_output_shapes

:2
model_12/dense_48/Relu?
'model_12/dense_49/MatMul/ReadVariableOpReadVariableOp0model_12_dense_49_matmul_readvariableop_resource*
_output_shapes

:*
dtype02)
'model_12/dense_49/MatMul/ReadVariableOp?
model_12/dense_49/MatMulMatMul$model_12/dense_48/Relu:activations:0/model_12/dense_49/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
model_12/dense_49/MatMul?
(model_12/dense_49/BiasAdd/ReadVariableOpReadVariableOp1model_12_dense_49_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(model_12/dense_49/BiasAdd/ReadVariableOp?
model_12/dense_49/BiasAddBiasAdd"model_12/dense_49/MatMul:product:00model_12/dense_49/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
model_12/dense_49/BiasAdd?
model_12/dense_49/ReluRelu"model_12/dense_49/BiasAdd:output:0*
T0*
_output_shapes

:2
model_12/dense_49/Relu?
'model_12/dense_50/MatMul/ReadVariableOpReadVariableOp0model_12_dense_50_matmul_readvariableop_resource*
_output_shapes

:*
dtype02)
'model_12/dense_50/MatMul/ReadVariableOp?
model_12/dense_50/MatMulMatMul$model_12/dense_49/Relu:activations:0/model_12/dense_50/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
model_12/dense_50/MatMul?
(model_12/dense_50/BiasAdd/ReadVariableOpReadVariableOp1model_12_dense_50_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(model_12/dense_50/BiasAdd/ReadVariableOp?
model_12/dense_50/BiasAddBiasAdd"model_12/dense_50/MatMul:product:00model_12/dense_50/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
model_12/dense_50/BiasAdd?
model_12/dense_50/ReluRelu"model_12/dense_50/BiasAdd:output:0*
T0*
_output_shapes

:2
model_12/dense_50/Relu?
'model_12/dense_51/MatMul/ReadVariableOpReadVariableOp0model_12_dense_51_matmul_readvariableop_resource*
_output_shapes

:*
dtype02)
'model_12/dense_51/MatMul/ReadVariableOp?
model_12/dense_51/MatMulMatMul$model_12/dense_50/Relu:activations:0/model_12/dense_51/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
model_12/dense_51/MatMul?
(model_12/dense_51/BiasAdd/ReadVariableOpReadVariableOp1model_12_dense_51_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(model_12/dense_51/BiasAdd/ReadVariableOp?
model_12/dense_51/BiasAddBiasAdd"model_12/dense_51/MatMul:product:00model_12/dense_51/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
model_12/dense_51/BiasAdd?
#model_12/tf.math.sigmoid_12/SigmoidSigmoid"model_12/dense_51/BiasAdd:output:0*
T0*
_output_shapes

:2%
#model_12/tf.math.sigmoid_12/Sigmoidy
IdentityIdentity'model_12/tf.math.sigmoid_12/Sigmoid:y:0^NoOp*
T0*
_output_shapes

:2

Identity?
NoOpNoOp,^model_12/attention_12/MatMul/ReadVariableOp.^model_12/attention_12/MatMul_3/ReadVariableOp)^model_12/dense_48/BiasAdd/ReadVariableOp(^model_12/dense_48/MatMul/ReadVariableOp)^model_12/dense_49/BiasAdd/ReadVariableOp(^model_12/dense_49/MatMul/ReadVariableOp)^model_12/dense_50/BiasAdd/ReadVariableOp(^model_12/dense_50/MatMul/ReadVariableOp)^model_12/dense_51/BiasAdd/ReadVariableOp(^model_12/dense_51/MatMul/ReadVariableOp,^model_12/graph_conv_36/add_1/ReadVariableOp,^model_12/graph_conv_36/add_5/ReadVariableOp0^model_12/graph_conv_36/transpose/ReadVariableOp2^model_12/graph_conv_36/transpose_2/ReadVariableOp,^model_12/graph_conv_37/add_1/ReadVariableOp,^model_12/graph_conv_37/add_5/ReadVariableOp0^model_12/graph_conv_37/transpose/ReadVariableOp2^model_12/graph_conv_37/transpose_2/ReadVariableOp,^model_12/graph_conv_38/add_1/ReadVariableOp,^model_12/graph_conv_38/add_5/ReadVariableOp0^model_12/graph_conv_38/transpose/ReadVariableOp2^model_12/graph_conv_38/transpose_2/ReadVariableOp6^model_12/neural_tensor_layer_12/MatMul/ReadVariableOp/^model_12/neural_tensor_layer_12/ReadVariableOp1^model_12/neural_tensor_layer_12/ReadVariableOp_12^model_12/neural_tensor_layer_12/ReadVariableOp_102^model_12/neural_tensor_layer_12/ReadVariableOp_112^model_12/neural_tensor_layer_12/ReadVariableOp_122^model_12/neural_tensor_layer_12/ReadVariableOp_132^model_12/neural_tensor_layer_12/ReadVariableOp_142^model_12/neural_tensor_layer_12/ReadVariableOp_151^model_12/neural_tensor_layer_12/ReadVariableOp_21^model_12/neural_tensor_layer_12/ReadVariableOp_31^model_12/neural_tensor_layer_12/ReadVariableOp_41^model_12/neural_tensor_layer_12/ReadVariableOp_51^model_12/neural_tensor_layer_12/ReadVariableOp_61^model_12/neural_tensor_layer_12/ReadVariableOp_71^model_12/neural_tensor_layer_12/ReadVariableOp_81^model_12/neural_tensor_layer_12/ReadVariableOp_93^model_12/neural_tensor_layer_12/add/ReadVariableOp5^model_12/neural_tensor_layer_12/add_1/ReadVariableOp6^model_12/neural_tensor_layer_12/add_10/ReadVariableOp6^model_12/neural_tensor_layer_12/add_11/ReadVariableOp6^model_12/neural_tensor_layer_12/add_12/ReadVariableOp6^model_12/neural_tensor_layer_12/add_13/ReadVariableOp6^model_12/neural_tensor_layer_12/add_14/ReadVariableOp6^model_12/neural_tensor_layer_12/add_15/ReadVariableOp5^model_12/neural_tensor_layer_12/add_2/ReadVariableOp5^model_12/neural_tensor_layer_12/add_3/ReadVariableOp5^model_12/neural_tensor_layer_12/add_4/ReadVariableOp5^model_12/neural_tensor_layer_12/add_5/ReadVariableOp5^model_12/neural_tensor_layer_12/add_6/ReadVariableOp5^model_12/neural_tensor_layer_12/add_7/ReadVariableOp5^model_12/neural_tensor_layer_12/add_8/ReadVariableOp5^model_12/neural_tensor_layer_12/add_9/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????	:'???????????????????????????:?????????	:'???????????????????????????: : : : : : : : : : : : : : : : : : 2Z
+model_12/attention_12/MatMul/ReadVariableOp+model_12/attention_12/MatMul/ReadVariableOp2^
-model_12/attention_12/MatMul_3/ReadVariableOp-model_12/attention_12/MatMul_3/ReadVariableOp2T
(model_12/dense_48/BiasAdd/ReadVariableOp(model_12/dense_48/BiasAdd/ReadVariableOp2R
'model_12/dense_48/MatMul/ReadVariableOp'model_12/dense_48/MatMul/ReadVariableOp2T
(model_12/dense_49/BiasAdd/ReadVariableOp(model_12/dense_49/BiasAdd/ReadVariableOp2R
'model_12/dense_49/MatMul/ReadVariableOp'model_12/dense_49/MatMul/ReadVariableOp2T
(model_12/dense_50/BiasAdd/ReadVariableOp(model_12/dense_50/BiasAdd/ReadVariableOp2R
'model_12/dense_50/MatMul/ReadVariableOp'model_12/dense_50/MatMul/ReadVariableOp2T
(model_12/dense_51/BiasAdd/ReadVariableOp(model_12/dense_51/BiasAdd/ReadVariableOp2R
'model_12/dense_51/MatMul/ReadVariableOp'model_12/dense_51/MatMul/ReadVariableOp2Z
+model_12/graph_conv_36/add_1/ReadVariableOp+model_12/graph_conv_36/add_1/ReadVariableOp2Z
+model_12/graph_conv_36/add_5/ReadVariableOp+model_12/graph_conv_36/add_5/ReadVariableOp2b
/model_12/graph_conv_36/transpose/ReadVariableOp/model_12/graph_conv_36/transpose/ReadVariableOp2f
1model_12/graph_conv_36/transpose_2/ReadVariableOp1model_12/graph_conv_36/transpose_2/ReadVariableOp2Z
+model_12/graph_conv_37/add_1/ReadVariableOp+model_12/graph_conv_37/add_1/ReadVariableOp2Z
+model_12/graph_conv_37/add_5/ReadVariableOp+model_12/graph_conv_37/add_5/ReadVariableOp2b
/model_12/graph_conv_37/transpose/ReadVariableOp/model_12/graph_conv_37/transpose/ReadVariableOp2f
1model_12/graph_conv_37/transpose_2/ReadVariableOp1model_12/graph_conv_37/transpose_2/ReadVariableOp2Z
+model_12/graph_conv_38/add_1/ReadVariableOp+model_12/graph_conv_38/add_1/ReadVariableOp2Z
+model_12/graph_conv_38/add_5/ReadVariableOp+model_12/graph_conv_38/add_5/ReadVariableOp2b
/model_12/graph_conv_38/transpose/ReadVariableOp/model_12/graph_conv_38/transpose/ReadVariableOp2f
1model_12/graph_conv_38/transpose_2/ReadVariableOp1model_12/graph_conv_38/transpose_2/ReadVariableOp2n
5model_12/neural_tensor_layer_12/MatMul/ReadVariableOp5model_12/neural_tensor_layer_12/MatMul/ReadVariableOp2`
.model_12/neural_tensor_layer_12/ReadVariableOp.model_12/neural_tensor_layer_12/ReadVariableOp2d
0model_12/neural_tensor_layer_12/ReadVariableOp_10model_12/neural_tensor_layer_12/ReadVariableOp_12f
1model_12/neural_tensor_layer_12/ReadVariableOp_101model_12/neural_tensor_layer_12/ReadVariableOp_102f
1model_12/neural_tensor_layer_12/ReadVariableOp_111model_12/neural_tensor_layer_12/ReadVariableOp_112f
1model_12/neural_tensor_layer_12/ReadVariableOp_121model_12/neural_tensor_layer_12/ReadVariableOp_122f
1model_12/neural_tensor_layer_12/ReadVariableOp_131model_12/neural_tensor_layer_12/ReadVariableOp_132f
1model_12/neural_tensor_layer_12/ReadVariableOp_141model_12/neural_tensor_layer_12/ReadVariableOp_142f
1model_12/neural_tensor_layer_12/ReadVariableOp_151model_12/neural_tensor_layer_12/ReadVariableOp_152d
0model_12/neural_tensor_layer_12/ReadVariableOp_20model_12/neural_tensor_layer_12/ReadVariableOp_22d
0model_12/neural_tensor_layer_12/ReadVariableOp_30model_12/neural_tensor_layer_12/ReadVariableOp_32d
0model_12/neural_tensor_layer_12/ReadVariableOp_40model_12/neural_tensor_layer_12/ReadVariableOp_42d
0model_12/neural_tensor_layer_12/ReadVariableOp_50model_12/neural_tensor_layer_12/ReadVariableOp_52d
0model_12/neural_tensor_layer_12/ReadVariableOp_60model_12/neural_tensor_layer_12/ReadVariableOp_62d
0model_12/neural_tensor_layer_12/ReadVariableOp_70model_12/neural_tensor_layer_12/ReadVariableOp_72d
0model_12/neural_tensor_layer_12/ReadVariableOp_80model_12/neural_tensor_layer_12/ReadVariableOp_82d
0model_12/neural_tensor_layer_12/ReadVariableOp_90model_12/neural_tensor_layer_12/ReadVariableOp_92h
2model_12/neural_tensor_layer_12/add/ReadVariableOp2model_12/neural_tensor_layer_12/add/ReadVariableOp2l
4model_12/neural_tensor_layer_12/add_1/ReadVariableOp4model_12/neural_tensor_layer_12/add_1/ReadVariableOp2n
5model_12/neural_tensor_layer_12/add_10/ReadVariableOp5model_12/neural_tensor_layer_12/add_10/ReadVariableOp2n
5model_12/neural_tensor_layer_12/add_11/ReadVariableOp5model_12/neural_tensor_layer_12/add_11/ReadVariableOp2n
5model_12/neural_tensor_layer_12/add_12/ReadVariableOp5model_12/neural_tensor_layer_12/add_12/ReadVariableOp2n
5model_12/neural_tensor_layer_12/add_13/ReadVariableOp5model_12/neural_tensor_layer_12/add_13/ReadVariableOp2n
5model_12/neural_tensor_layer_12/add_14/ReadVariableOp5model_12/neural_tensor_layer_12/add_14/ReadVariableOp2n
5model_12/neural_tensor_layer_12/add_15/ReadVariableOp5model_12/neural_tensor_layer_12/add_15/ReadVariableOp2l
4model_12/neural_tensor_layer_12/add_2/ReadVariableOp4model_12/neural_tensor_layer_12/add_2/ReadVariableOp2l
4model_12/neural_tensor_layer_12/add_3/ReadVariableOp4model_12/neural_tensor_layer_12/add_3/ReadVariableOp2l
4model_12/neural_tensor_layer_12/add_4/ReadVariableOp4model_12/neural_tensor_layer_12/add_4/ReadVariableOp2l
4model_12/neural_tensor_layer_12/add_5/ReadVariableOp4model_12/neural_tensor_layer_12/add_5/ReadVariableOp2l
4model_12/neural_tensor_layer_12/add_6/ReadVariableOp4model_12/neural_tensor_layer_12/add_6/ReadVariableOp2l
4model_12/neural_tensor_layer_12/add_7/ReadVariableOp4model_12/neural_tensor_layer_12/add_7/ReadVariableOp2l
4model_12/neural_tensor_layer_12/add_8/ReadVariableOp4model_12/neural_tensor_layer_12/add_8/ReadVariableOp2l
4model_12/neural_tensor_layer_12/add_9/ReadVariableOp4model_12/neural_tensor_layer_12/add_9/ReadVariableOp:U Q
+
_output_shapes
:?????????	
"
_user_specified_name
input_49:gc
=
_output_shapes+
):'???????????????????????????
"
_user_specified_name
input_50:UQ
+
_output_shapes
:?????????	
"
_user_specified_name
input_51:gc
=
_output_shapes+
):'???????????????????????????
"
_user_specified_name
input_52
?
?
*__inference_dense_50_layer_call_fn_4265537

inputs
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_50_layer_call_and_return_conditional_losses_42629882
StatefulPartitionedCallr
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
:: : 22
StatefulPartitionedCallStatefulPartitionedCall:F B

_output_shapes

:
 
_user_specified_nameinputs
?
?
I__inference_attention_12_layer_call_and_return_conditional_losses_4265276
	embedding0
matmul_readvariableop_resource:
identity??MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOpv
MatMulMatMul	embeddingMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMulr
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2
Mean/reduction_indicesl
MeanMeanMatMul:product:0Mean/reduction_indices:output:0*
T0*
_output_shapes
:2
MeanH
TanhTanhMean:output:0*
T0*
_output_shapes
:2
Tanho
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ????2
Reshape/shapeh
ReshapeReshapeTanh:y:0Reshape/shape:output:0*
T0*
_output_shapes

:2	
Reshapeq
transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose/permw
	transpose	TransposeReshape:output:0transpose/perm:output:0*
T0*
_output_shapes

:2
	transposej
MatMul_1MatMul	embeddingtranspose:y:0*
T0*'
_output_shapes
:?????????2

MatMul_1c
SigmoidSigmoidMatMul_1:product:0*
T0*'
_output_shapes
:?????????2	
Sigmoidu
transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_1/perm
transpose_1	Transpose	embeddingtranspose_1/perm:output:0*
T0*'
_output_shapes
:?????????2
transpose_1e
MatMul_2MatMultranspose_1:y:0Sigmoid:y:0*
T0*
_output_shapes

:2

MatMul_2u
transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose_2/perm
transpose_2	TransposeMatMul_2:product:0transpose_2/perm:output:0*
T0*
_output_shapes

:2
transpose_2a
IdentityIdentitytranspose_2:y:0^NoOp*
T0*
_output_shapes

:2

Identityf
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:R N
'
_output_shapes
:?????????
#
_user_specified_name	embedding
?

?
E__inference_dense_50_layer_call_and_return_conditional_losses_4265548

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOpj
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpx
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2	
BiasAddO
ReluReluBiasAdd:output:0*
T0*
_output_shapes

:2
Relud
IdentityIdentityRelu:activations:0^NoOp*
T0*
_output_shapes

:2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
:: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:F B

_output_shapes

:
 
_user_specified_nameinputs
??
?
S__inference_neural_tensor_layer_12_layer_call_and_return_conditional_losses_4265488
inputs_0
inputs_10
matmul_readvariableop_resource: -
readvariableop_resource:)
add_readvariableop_resource:
identity??MatMul/ReadVariableOp?ReadVariableOp?ReadVariableOp_1?ReadVariableOp_10?ReadVariableOp_11?ReadVariableOp_12?ReadVariableOp_13?ReadVariableOp_14?ReadVariableOp_15?ReadVariableOp_2?ReadVariableOp_3?ReadVariableOp_4?ReadVariableOp_5?ReadVariableOp_6?ReadVariableOp_7?ReadVariableOp_8?ReadVariableOp_9?add/ReadVariableOp?add_1/ReadVariableOp?add_10/ReadVariableOp?add_11/ReadVariableOp?add_12/ReadVariableOp?add_13/ReadVariableOp?add_14/ReadVariableOp?add_15/ReadVariableOp?add_2/ReadVariableOp?add_3/ReadVariableOp?add_4/ReadVariableOp?add_5/ReadVariableOp?add_6/ReadVariableOp?add_7/ReadVariableOp?add_8/ReadVariableOp?add_9/ReadVariableOp_
ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisx
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*
_output_shapes

: 2
concat?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulconcat:output:0MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
MatMul|
ReadVariableOpReadVariableOpreadvariableop_resource*"
_output_shapes
:*
dtype02
ReadVariableOpx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceReadVariableOp:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
strided_slice_1k
MatMul_1MatMulinputs_0strided_slice_1:output:0*
T0*
_output_shapes

:2

MatMul_1X
mulMulinputs_1MatMul_1:product:0*
T0*
_output_shapes

:2
mul?
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:*
dtype02
add/ReadVariableOpa
addAddV2mul:z:0add/ReadVariableOp:value:0*
T0*
_output_shapes

:2
addp
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indices_
SumSumadd:z:0Sum/reduction_indices:output:0*
T0*
_output_shapes
:2
Sum?
ReadVariableOp_1ReadVariableOpreadvariableop_resource*"
_output_shapes
:*
dtype02
ReadVariableOp_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceReadVariableOp_1:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
strided_slice_2k
MatMul_2MatMulinputs_0strided_slice_2:output:0*
T0*
_output_shapes

:2

MatMul_2\
mul_1Mulinputs_1MatMul_2:product:0*
T0*
_output_shapes

:2
mul_1?
add_1/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:*
dtype02
add_1/ReadVariableOpi
add_1AddV2	mul_1:z:0add_1/ReadVariableOp:value:0*
T0*
_output_shapes

:2
add_1t
Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum_1/reduction_indicesg
Sum_1Sum	add_1:z:0 Sum_1/reduction_indices:output:0*
T0*
_output_shapes
:2
Sum_1?
ReadVariableOp_2ReadVariableOpreadvariableop_resource*"
_output_shapes
:*
dtype02
ReadVariableOp_2x
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSliceReadVariableOp_2:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
strided_slice_3k
MatMul_3MatMulinputs_0strided_slice_3:output:0*
T0*
_output_shapes

:2

MatMul_3\
mul_2Mulinputs_1MatMul_3:product:0*
T0*
_output_shapes

:2
mul_2?
add_2/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:*
dtype02
add_2/ReadVariableOpi
add_2AddV2	mul_2:z:0add_2/ReadVariableOp:value:0*
T0*
_output_shapes

:2
add_2t
Sum_2/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum_2/reduction_indicesg
Sum_2Sum	add_2:z:0 Sum_2/reduction_indices:output:0*
T0*
_output_shapes
:2
Sum_2?
ReadVariableOp_3ReadVariableOpreadvariableop_resource*"
_output_shapes
:*
dtype02
ReadVariableOp_3x
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_4/stack|
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_4/stack_1|
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_4/stack_2?
strided_slice_4StridedSliceReadVariableOp_3:value:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
strided_slice_4k
MatMul_4MatMulinputs_0strided_slice_4:output:0*
T0*
_output_shapes

:2

MatMul_4\
mul_3Mulinputs_1MatMul_4:product:0*
T0*
_output_shapes

:2
mul_3?
add_3/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:*
dtype02
add_3/ReadVariableOpi
add_3AddV2	mul_3:z:0add_3/ReadVariableOp:value:0*
T0*
_output_shapes

:2
add_3t
Sum_3/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum_3/reduction_indicesg
Sum_3Sum	add_3:z:0 Sum_3/reduction_indices:output:0*
T0*
_output_shapes
:2
Sum_3?
ReadVariableOp_4ReadVariableOpreadvariableop_resource*"
_output_shapes
:*
dtype02
ReadVariableOp_4x
strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_5/stack|
strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_5/stack_1|
strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_5/stack_2?
strided_slice_5StridedSliceReadVariableOp_4:value:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
strided_slice_5k
MatMul_5MatMulinputs_0strided_slice_5:output:0*
T0*
_output_shapes

:2

MatMul_5\
mul_4Mulinputs_1MatMul_5:product:0*
T0*
_output_shapes

:2
mul_4?
add_4/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:*
dtype02
add_4/ReadVariableOpi
add_4AddV2	mul_4:z:0add_4/ReadVariableOp:value:0*
T0*
_output_shapes

:2
add_4t
Sum_4/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum_4/reduction_indicesg
Sum_4Sum	add_4:z:0 Sum_4/reduction_indices:output:0*
T0*
_output_shapes
:2
Sum_4?
ReadVariableOp_5ReadVariableOpreadvariableop_resource*"
_output_shapes
:*
dtype02
ReadVariableOp_5x
strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_6/stack|
strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_6/stack_1|
strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_6/stack_2?
strided_slice_6StridedSliceReadVariableOp_5:value:0strided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
strided_slice_6k
MatMul_6MatMulinputs_0strided_slice_6:output:0*
T0*
_output_shapes

:2

MatMul_6\
mul_5Mulinputs_1MatMul_6:product:0*
T0*
_output_shapes

:2
mul_5?
add_5/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:*
dtype02
add_5/ReadVariableOpi
add_5AddV2	mul_5:z:0add_5/ReadVariableOp:value:0*
T0*
_output_shapes

:2
add_5t
Sum_5/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum_5/reduction_indicesg
Sum_5Sum	add_5:z:0 Sum_5/reduction_indices:output:0*
T0*
_output_shapes
:2
Sum_5?
ReadVariableOp_6ReadVariableOpreadvariableop_resource*"
_output_shapes
:*
dtype02
ReadVariableOp_6x
strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_7/stack|
strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_7/stack_1|
strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_7/stack_2?
strided_slice_7StridedSliceReadVariableOp_6:value:0strided_slice_7/stack:output:0 strided_slice_7/stack_1:output:0 strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
strided_slice_7k
MatMul_7MatMulinputs_0strided_slice_7:output:0*
T0*
_output_shapes

:2

MatMul_7\
mul_6Mulinputs_1MatMul_7:product:0*
T0*
_output_shapes

:2
mul_6?
add_6/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:*
dtype02
add_6/ReadVariableOpi
add_6AddV2	mul_6:z:0add_6/ReadVariableOp:value:0*
T0*
_output_shapes

:2
add_6t
Sum_6/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum_6/reduction_indicesg
Sum_6Sum	add_6:z:0 Sum_6/reduction_indices:output:0*
T0*
_output_shapes
:2
Sum_6?
ReadVariableOp_7ReadVariableOpreadvariableop_resource*"
_output_shapes
:*
dtype02
ReadVariableOp_7x
strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_8/stack|
strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_8/stack_1|
strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_8/stack_2?
strided_slice_8StridedSliceReadVariableOp_7:value:0strided_slice_8/stack:output:0 strided_slice_8/stack_1:output:0 strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
strided_slice_8k
MatMul_8MatMulinputs_0strided_slice_8:output:0*
T0*
_output_shapes

:2

MatMul_8\
mul_7Mulinputs_1MatMul_8:product:0*
T0*
_output_shapes

:2
mul_7?
add_7/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:*
dtype02
add_7/ReadVariableOpi
add_7AddV2	mul_7:z:0add_7/ReadVariableOp:value:0*
T0*
_output_shapes

:2
add_7t
Sum_7/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum_7/reduction_indicesg
Sum_7Sum	add_7:z:0 Sum_7/reduction_indices:output:0*
T0*
_output_shapes
:2
Sum_7?
ReadVariableOp_8ReadVariableOpreadvariableop_resource*"
_output_shapes
:*
dtype02
ReadVariableOp_8x
strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_9/stack|
strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:	2
strided_slice_9/stack_1|
strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_9/stack_2?
strided_slice_9StridedSliceReadVariableOp_8:value:0strided_slice_9/stack:output:0 strided_slice_9/stack_1:output:0 strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
strided_slice_9k
MatMul_9MatMulinputs_0strided_slice_9:output:0*
T0*
_output_shapes

:2

MatMul_9\
mul_8Mulinputs_1MatMul_9:product:0*
T0*
_output_shapes

:2
mul_8?
add_8/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:*
dtype02
add_8/ReadVariableOpi
add_8AddV2	mul_8:z:0add_8/ReadVariableOp:value:0*
T0*
_output_shapes

:2
add_8t
Sum_8/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum_8/reduction_indicesg
Sum_8Sum	add_8:z:0 Sum_8/reduction_indices:output:0*
T0*
_output_shapes
:2
Sum_8?
ReadVariableOp_9ReadVariableOpreadvariableop_resource*"
_output_shapes
:*
dtype02
ReadVariableOp_9z
strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB:	2
strided_slice_10/stack~
strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
2
strided_slice_10/stack_1~
strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_10/stack_2?
strided_slice_10StridedSliceReadVariableOp_9:value:0strided_slice_10/stack:output:0!strided_slice_10/stack_1:output:0!strided_slice_10/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
strided_slice_10n
	MatMul_10MatMulinputs_0strided_slice_10:output:0*
T0*
_output_shapes

:2
	MatMul_10]
mul_9Mulinputs_1MatMul_10:product:0*
T0*
_output_shapes

:2
mul_9?
add_9/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:*
dtype02
add_9/ReadVariableOpi
add_9AddV2	mul_9:z:0add_9/ReadVariableOp:value:0*
T0*
_output_shapes

:2
add_9t
Sum_9/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum_9/reduction_indicesg
Sum_9Sum	add_9:z:0 Sum_9/reduction_indices:output:0*
T0*
_output_shapes
:2
Sum_9?
ReadVariableOp_10ReadVariableOpreadvariableop_resource*"
_output_shapes
:*
dtype02
ReadVariableOp_10z
strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:
2
strided_slice_11/stack~
strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_11/stack_1~
strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_11/stack_2?
strided_slice_11StridedSliceReadVariableOp_10:value:0strided_slice_11/stack:output:0!strided_slice_11/stack_1:output:0!strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
strided_slice_11n
	MatMul_11MatMulinputs_0strided_slice_11:output:0*
T0*
_output_shapes

:2
	MatMul_11_
mul_10Mulinputs_1MatMul_11:product:0*
T0*
_output_shapes

:2
mul_10?
add_10/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:*
dtype02
add_10/ReadVariableOpm
add_10AddV2
mul_10:z:0add_10/ReadVariableOp:value:0*
T0*
_output_shapes

:2
add_10v
Sum_10/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum_10/reduction_indicesk
Sum_10Sum
add_10:z:0!Sum_10/reduction_indices:output:0*
T0*
_output_shapes
:2
Sum_10?
ReadVariableOp_11ReadVariableOpreadvariableop_resource*"
_output_shapes
:*
dtype02
ReadVariableOp_11z
strided_slice_12/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_12/stack~
strided_slice_12/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_12/stack_1~
strided_slice_12/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_12/stack_2?
strided_slice_12StridedSliceReadVariableOp_11:value:0strided_slice_12/stack:output:0!strided_slice_12/stack_1:output:0!strided_slice_12/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
strided_slice_12n
	MatMul_12MatMulinputs_0strided_slice_12:output:0*
T0*
_output_shapes

:2
	MatMul_12_
mul_11Mulinputs_1MatMul_12:product:0*
T0*
_output_shapes

:2
mul_11?
add_11/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:*
dtype02
add_11/ReadVariableOpm
add_11AddV2
mul_11:z:0add_11/ReadVariableOp:value:0*
T0*
_output_shapes

:2
add_11v
Sum_11/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum_11/reduction_indicesk
Sum_11Sum
add_11:z:0!Sum_11/reduction_indices:output:0*
T0*
_output_shapes
:2
Sum_11?
ReadVariableOp_12ReadVariableOpreadvariableop_resource*"
_output_shapes
:*
dtype02
ReadVariableOp_12z
strided_slice_13/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_13/stack~
strided_slice_13/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_13/stack_1~
strided_slice_13/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_13/stack_2?
strided_slice_13StridedSliceReadVariableOp_12:value:0strided_slice_13/stack:output:0!strided_slice_13/stack_1:output:0!strided_slice_13/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
strided_slice_13n
	MatMul_13MatMulinputs_0strided_slice_13:output:0*
T0*
_output_shapes

:2
	MatMul_13_
mul_12Mulinputs_1MatMul_13:product:0*
T0*
_output_shapes

:2
mul_12?
add_12/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:*
dtype02
add_12/ReadVariableOpm
add_12AddV2
mul_12:z:0add_12/ReadVariableOp:value:0*
T0*
_output_shapes

:2
add_12v
Sum_12/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum_12/reduction_indicesk
Sum_12Sum
add_12:z:0!Sum_12/reduction_indices:output:0*
T0*
_output_shapes
:2
Sum_12?
ReadVariableOp_13ReadVariableOpreadvariableop_resource*"
_output_shapes
:*
dtype02
ReadVariableOp_13z
strided_slice_14/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_14/stack~
strided_slice_14/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_14/stack_1~
strided_slice_14/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_14/stack_2?
strided_slice_14StridedSliceReadVariableOp_13:value:0strided_slice_14/stack:output:0!strided_slice_14/stack_1:output:0!strided_slice_14/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
strided_slice_14n
	MatMul_14MatMulinputs_0strided_slice_14:output:0*
T0*
_output_shapes

:2
	MatMul_14_
mul_13Mulinputs_1MatMul_14:product:0*
T0*
_output_shapes

:2
mul_13?
add_13/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:*
dtype02
add_13/ReadVariableOpm
add_13AddV2
mul_13:z:0add_13/ReadVariableOp:value:0*
T0*
_output_shapes

:2
add_13v
Sum_13/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum_13/reduction_indicesk
Sum_13Sum
add_13:z:0!Sum_13/reduction_indices:output:0*
T0*
_output_shapes
:2
Sum_13?
ReadVariableOp_14ReadVariableOpreadvariableop_resource*"
_output_shapes
:*
dtype02
ReadVariableOp_14z
strided_slice_15/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_15/stack~
strided_slice_15/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_15/stack_1~
strided_slice_15/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_15/stack_2?
strided_slice_15StridedSliceReadVariableOp_14:value:0strided_slice_15/stack:output:0!strided_slice_15/stack_1:output:0!strided_slice_15/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
strided_slice_15n
	MatMul_15MatMulinputs_0strided_slice_15:output:0*
T0*
_output_shapes

:2
	MatMul_15_
mul_14Mulinputs_1MatMul_15:product:0*
T0*
_output_shapes

:2
mul_14?
add_14/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:*
dtype02
add_14/ReadVariableOpm
add_14AddV2
mul_14:z:0add_14/ReadVariableOp:value:0*
T0*
_output_shapes

:2
add_14v
Sum_14/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum_14/reduction_indicesk
Sum_14Sum
add_14:z:0!Sum_14/reduction_indices:output:0*
T0*
_output_shapes
:2
Sum_14?
ReadVariableOp_15ReadVariableOpreadvariableop_resource*"
_output_shapes
:*
dtype02
ReadVariableOp_15z
strided_slice_16/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_16/stack~
strided_slice_16/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_16/stack_1~
strided_slice_16/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_16/stack_2?
strided_slice_16StridedSliceReadVariableOp_15:value:0strided_slice_16/stack:output:0!strided_slice_16/stack_1:output:0!strided_slice_16/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
strided_slice_16n
	MatMul_16MatMulinputs_0strided_slice_16:output:0*
T0*
_output_shapes

:2
	MatMul_16_
mul_15Mulinputs_1MatMul_16:product:0*
T0*
_output_shapes

:2
mul_15?
add_15/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:*
dtype02
add_15/ReadVariableOpm
add_15AddV2
mul_15:z:0add_15/ReadVariableOp:value:0*
T0*
_output_shapes

:2
add_15v
Sum_15/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum_15/reduction_indicesk
Sum_15Sum
add_15:z:0!Sum_15/reduction_indices:output:0*
T0*
_output_shapes
:2
Sum_15`
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_1/axis?
concat_1ConcatV2Sum:output:0Sum_1:output:0Sum_2:output:0Sum_3:output:0Sum_4:output:0Sum_5:output:0Sum_6:output:0Sum_7:output:0Sum_8:output:0Sum_9:output:0Sum_10:output:0Sum_11:output:0Sum_12:output:0Sum_13:output:0Sum_14:output:0Sum_15:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes
:2

concat_1d
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapeq
ReshapeReshapeconcat_1:output:0Reshape/shape:output:0*
T0*
_output_shapes

:2	
Reshapef
add_16AddV2Reshape:output:0MatMul:product:0*
T0*
_output_shapes

:2
add_16I
TanhTanh
add_16:z:0*
T0*
_output_shapes

:2
TanhZ
IdentityIdentityTanh:y:0^NoOp*
T0*
_output_shapes

:2

Identity?
NoOpNoOp^MatMul/ReadVariableOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_10^ReadVariableOp_11^ReadVariableOp_12^ReadVariableOp_13^ReadVariableOp_14^ReadVariableOp_15^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8^ReadVariableOp_9^add/ReadVariableOp^add_1/ReadVariableOp^add_10/ReadVariableOp^add_11/ReadVariableOp^add_12/ReadVariableOp^add_13/ReadVariableOp^add_14/ReadVariableOp^add_15/ReadVariableOp^add_2/ReadVariableOp^add_3/ReadVariableOp^add_4/ReadVariableOp^add_5/ReadVariableOp^add_6/ReadVariableOp^add_7/ReadVariableOp^add_8/ReadVariableOp^add_9/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
::: : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12&
ReadVariableOp_10ReadVariableOp_102&
ReadVariableOp_11ReadVariableOp_112&
ReadVariableOp_12ReadVariableOp_122&
ReadVariableOp_13ReadVariableOp_132&
ReadVariableOp_14ReadVariableOp_142&
ReadVariableOp_15ReadVariableOp_152$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32$
ReadVariableOp_4ReadVariableOp_42$
ReadVariableOp_5ReadVariableOp_52$
ReadVariableOp_6ReadVariableOp_62$
ReadVariableOp_7ReadVariableOp_72$
ReadVariableOp_8ReadVariableOp_82$
ReadVariableOp_9ReadVariableOp_92(
add/ReadVariableOpadd/ReadVariableOp2,
add_1/ReadVariableOpadd_1/ReadVariableOp2.
add_10/ReadVariableOpadd_10/ReadVariableOp2.
add_11/ReadVariableOpadd_11/ReadVariableOp2.
add_12/ReadVariableOpadd_12/ReadVariableOp2.
add_13/ReadVariableOpadd_13/ReadVariableOp2.
add_14/ReadVariableOpadd_14/ReadVariableOp2.
add_15/ReadVariableOpadd_15/ReadVariableOp2,
add_2/ReadVariableOpadd_2/ReadVariableOp2,
add_3/ReadVariableOpadd_3/ReadVariableOp2,
add_4/ReadVariableOpadd_4/ReadVariableOp2,
add_5/ReadVariableOpadd_5/ReadVariableOp2,
add_6/ReadVariableOpadd_6/ReadVariableOp2,
add_7/ReadVariableOpadd_7/ReadVariableOp2,
add_8/ReadVariableOpadd_8/ReadVariableOp2,
add_9/ReadVariableOpadd_9/ReadVariableOp:H D

_output_shapes

:
"
_user_specified_name
inputs/0:HD

_output_shapes

:
"
_user_specified_name
inputs/1
?	
?
/__inference_graph_conv_36_layer_call_fn_4264896
inputs_0
inputs_1
unknown:@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_graph_conv_36_layer_call_and_return_conditional_losses_42625732
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:?????????	:'???????????????????????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:?????????	
"
_user_specified_name
inputs/0:gc
=
_output_shapes+
):'???????????????????????????
"
_user_specified_name
inputs/1
?	
?
/__inference_graph_conv_38_layer_call_fn_4265138
inputs_0
inputs_1
unknown: 
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_graph_conv_38_layer_call_and_return_conditional_losses_42626912
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*`
_input_shapesO
M:?????????????????? :'???????????????????????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :?????????????????? 
"
_user_specified_name
inputs/0:gc
=
_output_shapes+
):'???????????????????????????
"
_user_specified_name
inputs/1
?,
?
J__inference_graph_conv_37_layer_call_and_return_conditional_losses_4265078
inputs_0
inputs_11
shape_2_readvariableop_resource:@ +
add_1_readvariableop_resource: 
identity??add_1/ReadVariableOp?transpose/ReadVariableOp}
MatMulBatchMatMulV2inputs_1inputs_1*
T0*=
_output_shapes+
):'???????????????????????????2
MatMulM
ShapeShapeMatMul:output:0*
T0*
_output_shapes
:2
Shapev
addAddV2MatMul:output:0inputs_1*
T0*=
_output_shapes+
):'???????????????????????????2
add[
	Greater/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
	Greater/y?
GreaterGreateradd:z:0Greater/y:output:0*
T0*=
_output_shapes+
):'???????????????????????????2	
Greaterx
CastCastGreater:z:0*

DstT0*

SrcT0
*=
_output_shapes+
):'???????????????????????????2
CastJ
Shape_1Shapeinputs_0*
T0*
_output_shapes
:2	
Shape_1^
unstackUnpackShape_1:output:0*
T0*
_output_shapes
: : : *	
num2	
unstack?
Shape_2/ReadVariableOpReadVariableOpshape_2_readvariableop_resource*
_output_shapes

:@ *
dtype02
Shape_2/ReadVariableOpc
Shape_2Const*
_output_shapes
:*
dtype0*
valueB"@       2	
Shape_2`
	unstack_1UnpackShape_2:output:0*
T0*
_output_shapes
: : *	
num2
	unstack_1o
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   2
Reshape/shapeq
ReshapeReshapeinputs_0Reshape/shape:output:0*
T0*'
_output_shapes
:?????????@2	
Reshape?
transpose/ReadVariableOpReadVariableOpshape_2_readvariableop_resource*
_output_shapes

:@ *
dtype02
transpose/ReadVariableOpq
transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose/perm?
	transpose	Transpose transpose/ReadVariableOp:value:0transpose/perm:output:0*
T0*
_output_shapes

:@ 2
	transposes
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"@   ????2
Reshape_1/shapes
	Reshape_1Reshapetranspose:y:0Reshape_1/shape:output:0*
T0*
_output_shapes

:@ 2
	Reshape_1v
MatMul_1MatMulReshape:output:0Reshape_1:output:0*
T0*'
_output_shapes
:????????? 2

MatMul_1h
Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B : 2
Reshape_2/shape/2?
Reshape_2/shapePackunstack:output:0unstack:output:1Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape_2/shape?
	Reshape_2ReshapeMatMul_1:product:0Reshape_2/shape:output:0*
T0*4
_output_shapes"
 :?????????????????? 2
	Reshape_2?
add_1/ReadVariableOpReadVariableOpadd_1_readvariableop_resource*
_output_shapes
: *
dtype02
add_1/ReadVariableOp?
add_1AddV2Reshape_2:output:0add_1/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :?????????????????? 2
add_1?
MatMul_2BatchMatMulV2Cast:y:0Cast:y:0*
T0*=
_output_shapes+
):'???????????????????????????2

MatMul_2S
Shape_3ShapeMatMul_2:output:0*
T0*
_output_shapes
:2	
Shape_3|
add_2AddV2MatMul_2:output:0Cast:y:0*
T0*=
_output_shapes+
):'???????????????????????????2
add_2_
Greater_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
Greater_1/y?
	Greater_1Greater	add_2:z:0Greater_1/y:output:0*
T0*=
_output_shapes+
):'???????????????????????????2
	Greater_1~
Cast_1CastGreater_1:z:0*

DstT0*

SrcT0
*=
_output_shapes+
):'???????????????????????????2
Cast_1y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose
Cast_1:y:0transpose_1/perm:output:0*
T0*=
_output_shapes+
):'???????????????????????????2
transpose_1?
MatMul_3BatchMatMulV2transpose_1:y:0	add_1:z:0*
T0*4
_output_shapes"
 :?????????????????? 2

MatMul_3S
Shape_4ShapeMatMul_3:output:0*
T0*
_output_shapes
:2	
Shape_4p
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indices?
SumSum
Cast_1:y:0Sum/reduction_indices:output:0*
T0*4
_output_shapes"
 :??????????????????*
	keep_dims(2
SumW
add_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32	
add_3/yv
add_3AddV2Sum:output:0add_3/y:output:0*
T0*4
_output_shapes"
 :??????????????????2
add_3z
truedivRealDivMatMul_3:output:0	add_3:z:0*
T0*4
_output_shapes"
 :?????????????????? 2	
truediv`
ReluRelutruediv:z:0*
T0*4
_output_shapes"
 :?????????????????? 2
Reluz
IdentityIdentityRelu:activations:0^NoOp*
T0*4
_output_shapes"
 :?????????????????? 2

Identity?
NoOpNoOp^add_1/ReadVariableOp^transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*`
_input_shapesO
M:??????????????????@:'???????????????????????????: : 2,
add_1/ReadVariableOpadd_1/ReadVariableOp24
transpose/ReadVariableOptranspose/ReadVariableOp:^ Z
4
_output_shapes"
 :??????????????????@
"
_user_specified_name
inputs/0:gc
=
_output_shapes+
):'???????????????????????????
"
_user_specified_name
inputs/1
??
?,
#__inference__traced_restore_4265975
file_prefix@
.assignvariableop_graph_conv_36_graph_conv_36_w:@>
0assignvariableop_1_graph_conv_36_graph_conv_36_b:@B
0assignvariableop_2_graph_conv_37_graph_conv_37_w:@ >
0assignvariableop_3_graph_conv_37_graph_conv_37_b: B
0assignvariableop_4_graph_conv_38_graph_conv_38_w: >
0assignvariableop_5_graph_conv_38_graph_conv_38_b:<
*assignvariableop_6_attention_12_att_weight:H
2assignvariableop_7_neural_tensor_layer_12_variable:F
4assignvariableop_8_neural_tensor_layer_12_variable_1: B
4assignvariableop_9_neural_tensor_layer_12_variable_2:5
#assignvariableop_10_dense_48_kernel:/
!assignvariableop_11_dense_48_bias:5
#assignvariableop_12_dense_49_kernel:/
!assignvariableop_13_dense_49_bias:5
#assignvariableop_14_dense_50_kernel:/
!assignvariableop_15_dense_50_bias:5
#assignvariableop_16_dense_51_kernel:/
!assignvariableop_17_dense_51_bias:+
!assignvariableop_18_adadelta_iter:	 ,
"assignvariableop_19_adadelta_decay: 4
*assignvariableop_20_adadelta_learning_rate: *
 assignvariableop_21_adadelta_rho: #
assignvariableop_22_total: #
assignvariableop_23_count: %
assignvariableop_24_total_1: %
assignvariableop_25_count_1: W
Eassignvariableop_26_adadelta_graph_conv_36_graph_conv_36_w_accum_grad:@S
Eassignvariableop_27_adadelta_graph_conv_36_graph_conv_36_b_accum_grad:@W
Eassignvariableop_28_adadelta_graph_conv_37_graph_conv_37_w_accum_grad:@ S
Eassignvariableop_29_adadelta_graph_conv_37_graph_conv_37_b_accum_grad: W
Eassignvariableop_30_adadelta_graph_conv_38_graph_conv_38_w_accum_grad: S
Eassignvariableop_31_adadelta_graph_conv_38_graph_conv_38_b_accum_grad:Q
?assignvariableop_32_adadelta_attention_12_att_weight_accum_grad:]
Gassignvariableop_33_adadelta_neural_tensor_layer_12_variable_accum_grad:[
Iassignvariableop_34_adadelta_neural_tensor_layer_12_variable_accum_grad_1: W
Iassignvariableop_35_adadelta_neural_tensor_layer_12_variable_accum_grad_2:I
7assignvariableop_36_adadelta_dense_48_kernel_accum_grad:C
5assignvariableop_37_adadelta_dense_48_bias_accum_grad:I
7assignvariableop_38_adadelta_dense_49_kernel_accum_grad:C
5assignvariableop_39_adadelta_dense_49_bias_accum_grad:I
7assignvariableop_40_adadelta_dense_50_kernel_accum_grad:C
5assignvariableop_41_adadelta_dense_50_bias_accum_grad:I
7assignvariableop_42_adadelta_dense_51_kernel_accum_grad:C
5assignvariableop_43_adadelta_dense_51_bias_accum_grad:V
Dassignvariableop_44_adadelta_graph_conv_36_graph_conv_36_w_accum_var:@R
Dassignvariableop_45_adadelta_graph_conv_36_graph_conv_36_b_accum_var:@V
Dassignvariableop_46_adadelta_graph_conv_37_graph_conv_37_w_accum_var:@ R
Dassignvariableop_47_adadelta_graph_conv_37_graph_conv_37_b_accum_var: V
Dassignvariableop_48_adadelta_graph_conv_38_graph_conv_38_w_accum_var: R
Dassignvariableop_49_adadelta_graph_conv_38_graph_conv_38_b_accum_var:P
>assignvariableop_50_adadelta_attention_12_att_weight_accum_var:\
Fassignvariableop_51_adadelta_neural_tensor_layer_12_variable_accum_var:Z
Hassignvariableop_52_adadelta_neural_tensor_layer_12_variable_accum_var_1: V
Hassignvariableop_53_adadelta_neural_tensor_layer_12_variable_accum_var_2:H
6assignvariableop_54_adadelta_dense_48_kernel_accum_var:B
4assignvariableop_55_adadelta_dense_48_bias_accum_var:H
6assignvariableop_56_adadelta_dense_49_kernel_accum_var:B
4assignvariableop_57_adadelta_dense_49_bias_accum_var:H
6assignvariableop_58_adadelta_dense_50_kernel_accum_var:B
4assignvariableop_59_adadelta_dense_50_bias_accum_var:H
6assignvariableop_60_adadelta_dense_51_kernel_accum_var:B
4assignvariableop_61_adadelta_dense_51_bias_accum_var:
identity_63??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_55?AssignVariableOp_56?AssignVariableOp_57?AssignVariableOp_58?AssignVariableOp_59?AssignVariableOp_6?AssignVariableOp_60?AssignVariableOp_61?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?'
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:?*
dtype0*?&
value?&B?&?B?layer_with_weights-0/graph_conv_36_W/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-0/graph_conv_36_b/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/graph_conv_37_W/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/graph_conv_37_b/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-2/graph_conv_38_W/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-2/graph_conv_38_b/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-3/att_weight/.ATTRIBUTES/VARIABLE_VALUEB1layer_with_weights-4/W/.ATTRIBUTES/VARIABLE_VALUEB1layer_with_weights-4/V/.ATTRIBUTES/VARIABLE_VALUEB1layer_with_weights-4/b/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBdlayer_with_weights-0/graph_conv_36_W/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBdlayer_with_weights-0/graph_conv_36_b/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBdlayer_with_weights-1/graph_conv_37_W/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBdlayer_with_weights-1/graph_conv_37_b/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBdlayer_with_weights-2/graph_conv_38_W/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBdlayer_with_weights-2/graph_conv_38_b/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEB_layer_with_weights-3/att_weight/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-4/W/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-4/V/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-4/b/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBclayer_with_weights-0/graph_conv_36_W/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBclayer_with_weights-0/graph_conv_36_b/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBclayer_with_weights-1/graph_conv_37_W/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBclayer_with_weights-1/graph_conv_37_b/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBclayer_with_weights-2/graph_conv_38_W/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBclayer_with_weights-2/graph_conv_38_b/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEB^layer_with_weights-3/att_weight/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-4/W/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-4/V/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-4/b/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:?*
dtype0*?
value?B??B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*M
dtypesC
A2?	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp.assignvariableop_graph_conv_36_graph_conv_36_wIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp0assignvariableop_1_graph_conv_36_graph_conv_36_bIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp0assignvariableop_2_graph_conv_37_graph_conv_37_wIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp0assignvariableop_3_graph_conv_37_graph_conv_37_bIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp0assignvariableop_4_graph_conv_38_graph_conv_38_wIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp0assignvariableop_5_graph_conv_38_graph_conv_38_bIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp*assignvariableop_6_attention_12_att_weightIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp2assignvariableop_7_neural_tensor_layer_12_variableIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp4assignvariableop_8_neural_tensor_layer_12_variable_1Identity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp4assignvariableop_9_neural_tensor_layer_12_variable_2Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp#assignvariableop_10_dense_48_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp!assignvariableop_11_dense_48_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp#assignvariableop_12_dense_49_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp!assignvariableop_13_dense_49_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp#assignvariableop_14_dense_50_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp!assignvariableop_15_dense_50_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp#assignvariableop_16_dense_51_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp!assignvariableop_17_dense_51_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp!assignvariableop_18_adadelta_iterIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp"assignvariableop_19_adadelta_decayIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp*assignvariableop_20_adadelta_learning_rateIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp assignvariableop_21_adadelta_rhoIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOpassignvariableop_22_totalIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOpassignvariableop_23_countIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOpassignvariableop_24_total_1Identity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOpassignvariableop_25_count_1Identity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOpEassignvariableop_26_adadelta_graph_conv_36_graph_conv_36_w_accum_gradIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOpEassignvariableop_27_adadelta_graph_conv_36_graph_conv_36_b_accum_gradIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOpEassignvariableop_28_adadelta_graph_conv_37_graph_conv_37_w_accum_gradIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOpEassignvariableop_29_adadelta_graph_conv_37_graph_conv_37_b_accum_gradIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOpEassignvariableop_30_adadelta_graph_conv_38_graph_conv_38_w_accum_gradIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOpEassignvariableop_31_adadelta_graph_conv_38_graph_conv_38_b_accum_gradIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp?assignvariableop_32_adadelta_attention_12_att_weight_accum_gradIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOpGassignvariableop_33_adadelta_neural_tensor_layer_12_variable_accum_gradIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOpIassignvariableop_34_adadelta_neural_tensor_layer_12_variable_accum_grad_1Identity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOpIassignvariableop_35_adadelta_neural_tensor_layer_12_variable_accum_grad_2Identity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp7assignvariableop_36_adadelta_dense_48_kernel_accum_gradIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp5assignvariableop_37_adadelta_dense_48_bias_accum_gradIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp7assignvariableop_38_adadelta_dense_49_kernel_accum_gradIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp5assignvariableop_39_adadelta_dense_49_bias_accum_gradIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp7assignvariableop_40_adadelta_dense_50_kernel_accum_gradIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOp5assignvariableop_41_adadelta_dense_50_bias_accum_gradIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOp7assignvariableop_42_adadelta_dense_51_kernel_accum_gradIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOp5assignvariableop_43_adadelta_dense_51_bias_accum_gradIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOpDassignvariableop_44_adadelta_graph_conv_36_graph_conv_36_w_accum_varIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOpDassignvariableop_45_adadelta_graph_conv_36_graph_conv_36_b_accum_varIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOpDassignvariableop_46_adadelta_graph_conv_37_graph_conv_37_w_accum_varIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOpDassignvariableop_47_adadelta_graph_conv_37_graph_conv_37_b_accum_varIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOpDassignvariableop_48_adadelta_graph_conv_38_graph_conv_38_w_accum_varIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49?
AssignVariableOp_49AssignVariableOpDassignvariableop_49_adadelta_graph_conv_38_graph_conv_38_b_accum_varIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50?
AssignVariableOp_50AssignVariableOp>assignvariableop_50_adadelta_attention_12_att_weight_accum_varIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51?
AssignVariableOp_51AssignVariableOpFassignvariableop_51_adadelta_neural_tensor_layer_12_variable_accum_varIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52?
AssignVariableOp_52AssignVariableOpHassignvariableop_52_adadelta_neural_tensor_layer_12_variable_accum_var_1Identity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53?
AssignVariableOp_53AssignVariableOpHassignvariableop_53_adadelta_neural_tensor_layer_12_variable_accum_var_2Identity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54?
AssignVariableOp_54AssignVariableOp6assignvariableop_54_adadelta_dense_48_kernel_accum_varIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55?
AssignVariableOp_55AssignVariableOp4assignvariableop_55_adadelta_dense_48_bias_accum_varIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56?
AssignVariableOp_56AssignVariableOp6assignvariableop_56_adadelta_dense_49_kernel_accum_varIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57?
AssignVariableOp_57AssignVariableOp4assignvariableop_57_adadelta_dense_49_bias_accum_varIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58?
AssignVariableOp_58AssignVariableOp6assignvariableop_58_adadelta_dense_50_kernel_accum_varIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59?
AssignVariableOp_59AssignVariableOp4assignvariableop_59_adadelta_dense_50_bias_accum_varIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60?
AssignVariableOp_60AssignVariableOp6assignvariableop_60_adadelta_dense_51_kernel_accum_varIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61?
AssignVariableOp_61AssignVariableOp4assignvariableop_61_adadelta_dense_51_bias_accum_varIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_619
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_62Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_62f
Identity_63IdentityIdentity_62:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_63?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"#
identity_63Identity_63:output:0*?
_input_shapes?
~: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?	
?
E__inference_dense_51_layer_call_and_return_conditional_losses_4265567

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOpj
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpx
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2	
BiasAddb
IdentityIdentityBiasAdd:output:0^NoOp*
T0*
_output_shapes

:2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
:: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:F B

_output_shapes

:
 
_user_specified_nameinputs
??
? 
 __inference__traced_save_4265779
file_prefix<
8savev2_graph_conv_36_graph_conv_36_w_read_readvariableop<
8savev2_graph_conv_36_graph_conv_36_b_read_readvariableop<
8savev2_graph_conv_37_graph_conv_37_w_read_readvariableop<
8savev2_graph_conv_37_graph_conv_37_b_read_readvariableop<
8savev2_graph_conv_38_graph_conv_38_w_read_readvariableop<
8savev2_graph_conv_38_graph_conv_38_b_read_readvariableop6
2savev2_attention_12_att_weight_read_readvariableop>
:savev2_neural_tensor_layer_12_variable_read_readvariableop@
<savev2_neural_tensor_layer_12_variable_1_read_readvariableop@
<savev2_neural_tensor_layer_12_variable_2_read_readvariableop.
*savev2_dense_48_kernel_read_readvariableop,
(savev2_dense_48_bias_read_readvariableop.
*savev2_dense_49_kernel_read_readvariableop,
(savev2_dense_49_bias_read_readvariableop.
*savev2_dense_50_kernel_read_readvariableop,
(savev2_dense_50_bias_read_readvariableop.
*savev2_dense_51_kernel_read_readvariableop,
(savev2_dense_51_bias_read_readvariableop,
(savev2_adadelta_iter_read_readvariableop	-
)savev2_adadelta_decay_read_readvariableop5
1savev2_adadelta_learning_rate_read_readvariableop+
'savev2_adadelta_rho_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableopP
Lsavev2_adadelta_graph_conv_36_graph_conv_36_w_accum_grad_read_readvariableopP
Lsavev2_adadelta_graph_conv_36_graph_conv_36_b_accum_grad_read_readvariableopP
Lsavev2_adadelta_graph_conv_37_graph_conv_37_w_accum_grad_read_readvariableopP
Lsavev2_adadelta_graph_conv_37_graph_conv_37_b_accum_grad_read_readvariableopP
Lsavev2_adadelta_graph_conv_38_graph_conv_38_w_accum_grad_read_readvariableopP
Lsavev2_adadelta_graph_conv_38_graph_conv_38_b_accum_grad_read_readvariableopJ
Fsavev2_adadelta_attention_12_att_weight_accum_grad_read_readvariableopR
Nsavev2_adadelta_neural_tensor_layer_12_variable_accum_grad_read_readvariableopT
Psavev2_adadelta_neural_tensor_layer_12_variable_accum_grad_1_read_readvariableopT
Psavev2_adadelta_neural_tensor_layer_12_variable_accum_grad_2_read_readvariableopB
>savev2_adadelta_dense_48_kernel_accum_grad_read_readvariableop@
<savev2_adadelta_dense_48_bias_accum_grad_read_readvariableopB
>savev2_adadelta_dense_49_kernel_accum_grad_read_readvariableop@
<savev2_adadelta_dense_49_bias_accum_grad_read_readvariableopB
>savev2_adadelta_dense_50_kernel_accum_grad_read_readvariableop@
<savev2_adadelta_dense_50_bias_accum_grad_read_readvariableopB
>savev2_adadelta_dense_51_kernel_accum_grad_read_readvariableop@
<savev2_adadelta_dense_51_bias_accum_grad_read_readvariableopO
Ksavev2_adadelta_graph_conv_36_graph_conv_36_w_accum_var_read_readvariableopO
Ksavev2_adadelta_graph_conv_36_graph_conv_36_b_accum_var_read_readvariableopO
Ksavev2_adadelta_graph_conv_37_graph_conv_37_w_accum_var_read_readvariableopO
Ksavev2_adadelta_graph_conv_37_graph_conv_37_b_accum_var_read_readvariableopO
Ksavev2_adadelta_graph_conv_38_graph_conv_38_w_accum_var_read_readvariableopO
Ksavev2_adadelta_graph_conv_38_graph_conv_38_b_accum_var_read_readvariableopI
Esavev2_adadelta_attention_12_att_weight_accum_var_read_readvariableopQ
Msavev2_adadelta_neural_tensor_layer_12_variable_accum_var_read_readvariableopS
Osavev2_adadelta_neural_tensor_layer_12_variable_accum_var_1_read_readvariableopS
Osavev2_adadelta_neural_tensor_layer_12_variable_accum_var_2_read_readvariableopA
=savev2_adadelta_dense_48_kernel_accum_var_read_readvariableop?
;savev2_adadelta_dense_48_bias_accum_var_read_readvariableopA
=savev2_adadelta_dense_49_kernel_accum_var_read_readvariableop?
;savev2_adadelta_dense_49_bias_accum_var_read_readvariableopA
=savev2_adadelta_dense_50_kernel_accum_var_read_readvariableop?
;savev2_adadelta_dense_50_bias_accum_var_read_readvariableopA
=savev2_adadelta_dense_51_kernel_accum_var_read_readvariableop?
;savev2_adadelta_dense_51_bias_accum_var_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?'
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:?*
dtype0*?&
value?&B?&?B?layer_with_weights-0/graph_conv_36_W/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-0/graph_conv_36_b/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/graph_conv_37_W/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/graph_conv_37_b/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-2/graph_conv_38_W/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-2/graph_conv_38_b/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-3/att_weight/.ATTRIBUTES/VARIABLE_VALUEB1layer_with_weights-4/W/.ATTRIBUTES/VARIABLE_VALUEB1layer_with_weights-4/V/.ATTRIBUTES/VARIABLE_VALUEB1layer_with_weights-4/b/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBdlayer_with_weights-0/graph_conv_36_W/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBdlayer_with_weights-0/graph_conv_36_b/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBdlayer_with_weights-1/graph_conv_37_W/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBdlayer_with_weights-1/graph_conv_37_b/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBdlayer_with_weights-2/graph_conv_38_W/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBdlayer_with_weights-2/graph_conv_38_b/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEB_layer_with_weights-3/att_weight/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-4/W/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-4/V/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-4/b/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEB[layer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBYlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/accum_grad/.ATTRIBUTES/VARIABLE_VALUEBclayer_with_weights-0/graph_conv_36_W/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBclayer_with_weights-0/graph_conv_36_b/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBclayer_with_weights-1/graph_conv_37_W/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBclayer_with_weights-1/graph_conv_37_b/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBclayer_with_weights-2/graph_conv_38_W/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBclayer_with_weights-2/graph_conv_38_b/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEB^layer_with_weights-3/att_weight/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-4/W/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-4/V/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-4/b/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBZlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEBXlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/accum_var/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:?*
dtype0*?
value?B??B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:08savev2_graph_conv_36_graph_conv_36_w_read_readvariableop8savev2_graph_conv_36_graph_conv_36_b_read_readvariableop8savev2_graph_conv_37_graph_conv_37_w_read_readvariableop8savev2_graph_conv_37_graph_conv_37_b_read_readvariableop8savev2_graph_conv_38_graph_conv_38_w_read_readvariableop8savev2_graph_conv_38_graph_conv_38_b_read_readvariableop2savev2_attention_12_att_weight_read_readvariableop:savev2_neural_tensor_layer_12_variable_read_readvariableop<savev2_neural_tensor_layer_12_variable_1_read_readvariableop<savev2_neural_tensor_layer_12_variable_2_read_readvariableop*savev2_dense_48_kernel_read_readvariableop(savev2_dense_48_bias_read_readvariableop*savev2_dense_49_kernel_read_readvariableop(savev2_dense_49_bias_read_readvariableop*savev2_dense_50_kernel_read_readvariableop(savev2_dense_50_bias_read_readvariableop*savev2_dense_51_kernel_read_readvariableop(savev2_dense_51_bias_read_readvariableop(savev2_adadelta_iter_read_readvariableop)savev2_adadelta_decay_read_readvariableop1savev2_adadelta_learning_rate_read_readvariableop'savev2_adadelta_rho_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableopLsavev2_adadelta_graph_conv_36_graph_conv_36_w_accum_grad_read_readvariableopLsavev2_adadelta_graph_conv_36_graph_conv_36_b_accum_grad_read_readvariableopLsavev2_adadelta_graph_conv_37_graph_conv_37_w_accum_grad_read_readvariableopLsavev2_adadelta_graph_conv_37_graph_conv_37_b_accum_grad_read_readvariableopLsavev2_adadelta_graph_conv_38_graph_conv_38_w_accum_grad_read_readvariableopLsavev2_adadelta_graph_conv_38_graph_conv_38_b_accum_grad_read_readvariableopFsavev2_adadelta_attention_12_att_weight_accum_grad_read_readvariableopNsavev2_adadelta_neural_tensor_layer_12_variable_accum_grad_read_readvariableopPsavev2_adadelta_neural_tensor_layer_12_variable_accum_grad_1_read_readvariableopPsavev2_adadelta_neural_tensor_layer_12_variable_accum_grad_2_read_readvariableop>savev2_adadelta_dense_48_kernel_accum_grad_read_readvariableop<savev2_adadelta_dense_48_bias_accum_grad_read_readvariableop>savev2_adadelta_dense_49_kernel_accum_grad_read_readvariableop<savev2_adadelta_dense_49_bias_accum_grad_read_readvariableop>savev2_adadelta_dense_50_kernel_accum_grad_read_readvariableop<savev2_adadelta_dense_50_bias_accum_grad_read_readvariableop>savev2_adadelta_dense_51_kernel_accum_grad_read_readvariableop<savev2_adadelta_dense_51_bias_accum_grad_read_readvariableopKsavev2_adadelta_graph_conv_36_graph_conv_36_w_accum_var_read_readvariableopKsavev2_adadelta_graph_conv_36_graph_conv_36_b_accum_var_read_readvariableopKsavev2_adadelta_graph_conv_37_graph_conv_37_w_accum_var_read_readvariableopKsavev2_adadelta_graph_conv_37_graph_conv_37_b_accum_var_read_readvariableopKsavev2_adadelta_graph_conv_38_graph_conv_38_w_accum_var_read_readvariableopKsavev2_adadelta_graph_conv_38_graph_conv_38_b_accum_var_read_readvariableopEsavev2_adadelta_attention_12_att_weight_accum_var_read_readvariableopMsavev2_adadelta_neural_tensor_layer_12_variable_accum_var_read_readvariableopOsavev2_adadelta_neural_tensor_layer_12_variable_accum_var_1_read_readvariableopOsavev2_adadelta_neural_tensor_layer_12_variable_accum_var_2_read_readvariableop=savev2_adadelta_dense_48_kernel_accum_var_read_readvariableop;savev2_adadelta_dense_48_bias_accum_var_read_readvariableop=savev2_adadelta_dense_49_kernel_accum_var_read_readvariableop;savev2_adadelta_dense_49_bias_accum_var_read_readvariableop=savev2_adadelta_dense_50_kernel_accum_var_read_readvariableop;savev2_adadelta_dense_50_bias_accum_var_read_readvariableop=savev2_adadelta_dense_51_kernel_accum_var_read_readvariableop;savev2_adadelta_dense_51_bias_accum_var_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *M
dtypesC
A2?	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_1c
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0*?
_input_shapes?
?: :@:@:@ : : :::: :::::::::: : : : : : : : :@:@:@ : : :::: ::::::::::@:@:@ : : :::: :::::::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:@: 

_output_shapes
:@:$ 

_output_shapes

:@ : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::$ 

_output_shapes

::($
"
_output_shapes
::$	 

_output_shapes

: : 


_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:@: 

_output_shapes
:@:$ 

_output_shapes

:@ : 

_output_shapes
: :$ 

_output_shapes

: :  

_output_shapes
::$! 

_output_shapes

::("$
"
_output_shapes
::$# 

_output_shapes

: : $

_output_shapes
::$% 

_output_shapes

:: &

_output_shapes
::$' 

_output_shapes

:: (

_output_shapes
::$) 

_output_shapes

:: *

_output_shapes
::$+ 

_output_shapes

:: ,

_output_shapes
::$- 

_output_shapes

:@: .

_output_shapes
:@:$/ 

_output_shapes

:@ : 0

_output_shapes
: :$1 

_output_shapes

: : 2

_output_shapes
::$3 

_output_shapes

::(4$
"
_output_shapes
::$5 

_output_shapes

: : 6

_output_shapes
::$7 

_output_shapes

:: 8

_output_shapes
::$9 

_output_shapes

:: :

_output_shapes
::$; 

_output_shapes

:: <

_output_shapes
::$= 

_output_shapes

:: >

_output_shapes
::?

_output_shapes
: 
?-
?
J__inference_graph_conv_36_layer_call_and_return_conditional_losses_4263319

inputs
inputs_11
shape_2_readvariableop_resource:@+
add_1_readvariableop_resource:@
identity??add_1/ReadVariableOp?transpose/ReadVariableOp}
MatMulBatchMatMulV2inputs_1inputs_1*
T0*=
_output_shapes+
):'???????????????????????????2
MatMulM
ShapeShapeMatMul:output:0*
T0*
_output_shapes
:2
Shapev
addAddV2MatMul:output:0inputs_1*
T0*=
_output_shapes+
):'???????????????????????????2
add[
	Greater/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
	Greater/y?
GreaterGreateradd:z:0Greater/y:output:0*
T0*=
_output_shapes+
):'???????????????????????????2	
Greaterx
CastCastGreater:z:0*

DstT0*

SrcT0
*=
_output_shapes+
):'???????????????????????????2
CastH
Shape_1Shapeinputs*
T0*
_output_shapes
:2	
Shape_1^
unstackUnpackShape_1:output:0*
T0*
_output_shapes
: : : *	
num2	
unstack?
Shape_2/ReadVariableOpReadVariableOpshape_2_readvariableop_resource*
_output_shapes

:@*
dtype02
Shape_2/ReadVariableOpc
Shape_2Const*
_output_shapes
:*
dtype0*
valueB"   @   2	
Shape_2`
	unstack_1UnpackShape_2:output:0*
T0*
_output_shapes
: : *	
num2
	unstack_1o
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape/shapeo
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:?????????2	
Reshape?
transpose/ReadVariableOpReadVariableOpshape_2_readvariableop_resource*
_output_shapes

:@*
dtype02
transpose/ReadVariableOpq
transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose/perm?
	transpose	Transpose transpose/ReadVariableOp:value:0transpose/perm:output:0*
T0*
_output_shapes

:@2
	transposes
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ????2
Reshape_1/shapes
	Reshape_1Reshapetranspose:y:0Reshape_1/shape:output:0*
T0*
_output_shapes

:@2
	Reshape_1v
MatMul_1MatMulReshape:output:0Reshape_1:output:0*
T0*'
_output_shapes
:?????????@2

MatMul_1h
Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :	2
Reshape_2/shape/1h
Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :@2
Reshape_2/shape/2?
Reshape_2/shapePackunstack:output:0Reshape_2/shape/1:output:0Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape_2/shape?
	Reshape_2ReshapeMatMul_1:product:0Reshape_2/shape:output:0*
T0*+
_output_shapes
:?????????	@2
	Reshape_2?
add_1/ReadVariableOpReadVariableOpadd_1_readvariableop_resource*
_output_shapes
:@*
dtype02
add_1/ReadVariableOp
add_1AddV2Reshape_2:output:0add_1/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????	@2
add_1?
MatMul_2BatchMatMulV2Cast:y:0Cast:y:0*
T0*=
_output_shapes+
):'???????????????????????????2

MatMul_2S
Shape_3ShapeMatMul_2:output:0*
T0*
_output_shapes
:2	
Shape_3|
add_2AddV2MatMul_2:output:0Cast:y:0*
T0*=
_output_shapes+
):'???????????????????????????2
add_2_
Greater_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
Greater_1/y?
	Greater_1Greater	add_2:z:0Greater_1/y:output:0*
T0*=
_output_shapes+
):'???????????????????????????2
	Greater_1~
Cast_1CastGreater_1:z:0*

DstT0*

SrcT0
*=
_output_shapes+
):'???????????????????????????2
Cast_1y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose
Cast_1:y:0transpose_1/perm:output:0*
T0*=
_output_shapes+
):'???????????????????????????2
transpose_1?
MatMul_3BatchMatMulV2transpose_1:y:0	add_1:z:0*
T0*4
_output_shapes"
 :??????????????????@2

MatMul_3S
Shape_4ShapeMatMul_3:output:0*
T0*
_output_shapes
:2	
Shape_4p
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indices?
SumSum
Cast_1:y:0Sum/reduction_indices:output:0*
T0*4
_output_shapes"
 :??????????????????*
	keep_dims(2
SumW
add_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32	
add_3/yv
add_3AddV2Sum:output:0add_3/y:output:0*
T0*4
_output_shapes"
 :??????????????????2
add_3z
truedivRealDivMatMul_3:output:0	add_3:z:0*
T0*4
_output_shapes"
 :??????????????????@2	
truediv`
ReluRelutruediv:z:0*
T0*4
_output_shapes"
 :??????????????????@2
Reluz
IdentityIdentityRelu:activations:0^NoOp*
T0*4
_output_shapes"
 :??????????????????@2

Identity?
NoOpNoOp^add_1/ReadVariableOp^transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:?????????	:'???????????????????????????: : 2,
add_1/ReadVariableOpadd_1/ReadVariableOp24
transpose/ReadVariableOptranspose/ReadVariableOp:S O
+
_output_shapes
:?????????	
 
_user_specified_nameinputs:ea
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?]
?

E__inference_model_12_layer_call_and_return_conditional_losses_4263012

inputs
inputs_1
inputs_2
inputs_3'
graph_conv_36_4262574:@#
graph_conv_36_4262576:@'
graph_conv_37_4262633:@ #
graph_conv_37_4262635: '
graph_conv_38_4262692: #
graph_conv_38_4262694:&
attention_12_4262730:0
neural_tensor_layer_12_4262936: 4
neural_tensor_layer_12_4262938:,
neural_tensor_layer_12_4262940:"
dense_48_4262955:
dense_48_4262957:"
dense_49_4262972:
dense_49_4262974:"
dense_50_4262989:
dense_50_4262991:"
dense_51_4263005:
dense_51_4263007:
identity??$attention_12/StatefulPartitionedCall?&attention_12/StatefulPartitionedCall_1? dense_48/StatefulPartitionedCall? dense_49/StatefulPartitionedCall? dense_50/StatefulPartitionedCall? dense_51/StatefulPartitionedCall?%graph_conv_36/StatefulPartitionedCall?'graph_conv_36/StatefulPartitionedCall_1?%graph_conv_37/StatefulPartitionedCall?'graph_conv_37/StatefulPartitionedCall_1?%graph_conv_38/StatefulPartitionedCall?'graph_conv_38/StatefulPartitionedCall_1?.neural_tensor_layer_12/StatefulPartitionedCall?
%graph_conv_36/StatefulPartitionedCallStatefulPartitionedCallinputs_2inputs_3graph_conv_36_4262574graph_conv_36_4262576*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_graph_conv_36_layer_call_and_return_conditional_losses_42625732'
%graph_conv_36/StatefulPartitionedCall?
'graph_conv_36/StatefulPartitionedCall_1StatefulPartitionedCallinputsinputs_1graph_conv_36_4262574graph_conv_36_4262576*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_graph_conv_36_layer_call_and_return_conditional_losses_42625732)
'graph_conv_36/StatefulPartitionedCall_1?
%graph_conv_37/StatefulPartitionedCallStatefulPartitionedCall.graph_conv_36/StatefulPartitionedCall:output:0inputs_3graph_conv_37_4262633graph_conv_37_4262635*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :?????????????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_graph_conv_37_layer_call_and_return_conditional_losses_42626322'
%graph_conv_37/StatefulPartitionedCall?
'graph_conv_37/StatefulPartitionedCall_1StatefulPartitionedCall0graph_conv_36/StatefulPartitionedCall_1:output:0inputs_1graph_conv_37_4262633graph_conv_37_4262635*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :?????????????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_graph_conv_37_layer_call_and_return_conditional_losses_42626322)
'graph_conv_37/StatefulPartitionedCall_1?
%graph_conv_38/StatefulPartitionedCallStatefulPartitionedCall.graph_conv_37/StatefulPartitionedCall:output:0inputs_3graph_conv_38_4262692graph_conv_38_4262694*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_graph_conv_38_layer_call_and_return_conditional_losses_42626912'
%graph_conv_38/StatefulPartitionedCall?
'graph_conv_38/StatefulPartitionedCall_1StatefulPartitionedCall0graph_conv_37/StatefulPartitionedCall_1:output:0inputs_1graph_conv_38_4262692graph_conv_38_4262694*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_graph_conv_38_layer_call_and_return_conditional_losses_42626912)
'graph_conv_38/StatefulPartitionedCall_1?
/tf.__operators__.getitem_25/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/tf.__operators__.getitem_25/strided_slice/stack?
1tf.__operators__.getitem_25/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1tf.__operators__.getitem_25/strided_slice/stack_1?
1tf.__operators__.getitem_25/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1tf.__operators__.getitem_25/strided_slice/stack_2?
)tf.__operators__.getitem_25/strided_sliceStridedSlice.graph_conv_38/StatefulPartitionedCall:output:08tf.__operators__.getitem_25/strided_slice/stack:output:0:tf.__operators__.getitem_25/strided_slice/stack_1:output:0:tf.__operators__.getitem_25/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2+
)tf.__operators__.getitem_25/strided_slice?
/tf.__operators__.getitem_24/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/tf.__operators__.getitem_24/strided_slice/stack?
1tf.__operators__.getitem_24/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1tf.__operators__.getitem_24/strided_slice/stack_1?
1tf.__operators__.getitem_24/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1tf.__operators__.getitem_24/strided_slice/stack_2?
)tf.__operators__.getitem_24/strided_sliceStridedSlice0graph_conv_38/StatefulPartitionedCall_1:output:08tf.__operators__.getitem_24/strided_slice/stack:output:0:tf.__operators__.getitem_24/strided_slice/stack_1:output:0:tf.__operators__.getitem_24/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2+
)tf.__operators__.getitem_24/strided_slice?
$attention_12/StatefulPartitionedCallStatefulPartitionedCall2tf.__operators__.getitem_24/strided_slice:output:0attention_12_4262730*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_attention_12_layer_call_and_return_conditional_losses_42627292&
$attention_12/StatefulPartitionedCall?
&attention_12/StatefulPartitionedCall_1StatefulPartitionedCall2tf.__operators__.getitem_25/strided_slice:output:0attention_12_4262730*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_attention_12_layer_call_and_return_conditional_losses_42627292(
&attention_12/StatefulPartitionedCall_1?
.neural_tensor_layer_12/StatefulPartitionedCallStatefulPartitionedCall-attention_12/StatefulPartitionedCall:output:0/attention_12/StatefulPartitionedCall_1:output:0neural_tensor_layer_12_4262936neural_tensor_layer_12_4262938neural_tensor_layer_12_4262940*
Tin	
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_neural_tensor_layer_12_layer_call_and_return_conditional_losses_426293520
.neural_tensor_layer_12/StatefulPartitionedCall?
 dense_48/StatefulPartitionedCallStatefulPartitionedCall7neural_tensor_layer_12/StatefulPartitionedCall:output:0dense_48_4262955dense_48_4262957*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_48_layer_call_and_return_conditional_losses_42629542"
 dense_48/StatefulPartitionedCall?
 dense_49/StatefulPartitionedCallStatefulPartitionedCall)dense_48/StatefulPartitionedCall:output:0dense_49_4262972dense_49_4262974*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_49_layer_call_and_return_conditional_losses_42629712"
 dense_49/StatefulPartitionedCall?
 dense_50/StatefulPartitionedCallStatefulPartitionedCall)dense_49/StatefulPartitionedCall:output:0dense_50_4262989dense_50_4262991*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_50_layer_call_and_return_conditional_losses_42629882"
 dense_50/StatefulPartitionedCall?
 dense_51/StatefulPartitionedCallStatefulPartitionedCall)dense_50/StatefulPartitionedCall:output:0dense_51_4263005dense_51_4263007*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_51_layer_call_and_return_conditional_losses_42630042"
 dense_51/StatefulPartitionedCall?
tf.math.sigmoid_12/SigmoidSigmoid)dense_51/StatefulPartitionedCall:output:0*
T0*
_output_shapes

:2
tf.math.sigmoid_12/Sigmoidp
IdentityIdentitytf.math.sigmoid_12/Sigmoid:y:0^NoOp*
T0*
_output_shapes

:2

Identity?
NoOpNoOp%^attention_12/StatefulPartitionedCall'^attention_12/StatefulPartitionedCall_1!^dense_48/StatefulPartitionedCall!^dense_49/StatefulPartitionedCall!^dense_50/StatefulPartitionedCall!^dense_51/StatefulPartitionedCall&^graph_conv_36/StatefulPartitionedCall(^graph_conv_36/StatefulPartitionedCall_1&^graph_conv_37/StatefulPartitionedCall(^graph_conv_37/StatefulPartitionedCall_1&^graph_conv_38/StatefulPartitionedCall(^graph_conv_38/StatefulPartitionedCall_1/^neural_tensor_layer_12/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????	:'???????????????????????????:?????????	:'???????????????????????????: : : : : : : : : : : : : : : : : : 2L
$attention_12/StatefulPartitionedCall$attention_12/StatefulPartitionedCall2P
&attention_12/StatefulPartitionedCall_1&attention_12/StatefulPartitionedCall_12D
 dense_48/StatefulPartitionedCall dense_48/StatefulPartitionedCall2D
 dense_49/StatefulPartitionedCall dense_49/StatefulPartitionedCall2D
 dense_50/StatefulPartitionedCall dense_50/StatefulPartitionedCall2D
 dense_51/StatefulPartitionedCall dense_51/StatefulPartitionedCall2N
%graph_conv_36/StatefulPartitionedCall%graph_conv_36/StatefulPartitionedCall2R
'graph_conv_36/StatefulPartitionedCall_1'graph_conv_36/StatefulPartitionedCall_12N
%graph_conv_37/StatefulPartitionedCall%graph_conv_37/StatefulPartitionedCall2R
'graph_conv_37/StatefulPartitionedCall_1'graph_conv_37/StatefulPartitionedCall_12N
%graph_conv_38/StatefulPartitionedCall%graph_conv_38/StatefulPartitionedCall2R
'graph_conv_38/StatefulPartitionedCall_1'graph_conv_38/StatefulPartitionedCall_12`
.neural_tensor_layer_12/StatefulPartitionedCall.neural_tensor_layer_12/StatefulPartitionedCall:S O
+
_output_shapes
:?????????	
 
_user_specified_nameinputs:ea
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs:SO
+
_output_shapes
:?????????	
 
_user_specified_nameinputs:ea
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
?
*__inference_dense_51_layer_call_fn_4265557

inputs
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_51_layer_call_and_return_conditional_losses_42630042
StatefulPartitionedCallr
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
:: : 22
StatefulPartitionedCallStatefulPartitionedCall:F B

_output_shapes

:
 
_user_specified_nameinputs
?,
?
J__inference_graph_conv_38_layer_call_and_return_conditional_losses_4263176

inputs
inputs_11
shape_2_readvariableop_resource: +
add_1_readvariableop_resource:
identity??add_1/ReadVariableOp?transpose/ReadVariableOp}
MatMulBatchMatMulV2inputs_1inputs_1*
T0*=
_output_shapes+
):'???????????????????????????2
MatMulM
ShapeShapeMatMul:output:0*
T0*
_output_shapes
:2
Shapev
addAddV2MatMul:output:0inputs_1*
T0*=
_output_shapes+
):'???????????????????????????2
add[
	Greater/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
	Greater/y?
GreaterGreateradd:z:0Greater/y:output:0*
T0*=
_output_shapes+
):'???????????????????????????2	
Greaterx
CastCastGreater:z:0*

DstT0*

SrcT0
*=
_output_shapes+
):'???????????????????????????2
CastH
Shape_1Shapeinputs*
T0*
_output_shapes
:2	
Shape_1^
unstackUnpackShape_1:output:0*
T0*
_output_shapes
: : : *	
num2	
unstack?
Shape_2/ReadVariableOpReadVariableOpshape_2_readvariableop_resource*
_output_shapes

: *
dtype02
Shape_2/ReadVariableOpc
Shape_2Const*
_output_shapes
:*
dtype0*
valueB"       2	
Shape_2`
	unstack_1UnpackShape_2:output:0*
T0*
_output_shapes
: : *	
num2
	unstack_1o
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????    2
Reshape/shapeo
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:????????? 2	
Reshape?
transpose/ReadVariableOpReadVariableOpshape_2_readvariableop_resource*
_output_shapes

: *
dtype02
transpose/ReadVariableOpq
transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose/perm?
	transpose	Transpose transpose/ReadVariableOp:value:0transpose/perm:output:0*
T0*
_output_shapes

: 2
	transposes
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"    ????2
Reshape_1/shapes
	Reshape_1Reshapetranspose:y:0Reshape_1/shape:output:0*
T0*
_output_shapes

: 2
	Reshape_1v
MatMul_1MatMulReshape:output:0Reshape_1:output:0*
T0*'
_output_shapes
:?????????2

MatMul_1h
Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape_2/shape/2?
Reshape_2/shapePackunstack:output:0unstack:output:1Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape_2/shape?
	Reshape_2ReshapeMatMul_1:product:0Reshape_2/shape:output:0*
T0*4
_output_shapes"
 :??????????????????2
	Reshape_2?
add_1/ReadVariableOpReadVariableOpadd_1_readvariableop_resource*
_output_shapes
:*
dtype02
add_1/ReadVariableOp?
add_1AddV2Reshape_2:output:0add_1/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :??????????????????2
add_1?
MatMul_2BatchMatMulV2Cast:y:0Cast:y:0*
T0*=
_output_shapes+
):'???????????????????????????2

MatMul_2S
Shape_3ShapeMatMul_2:output:0*
T0*
_output_shapes
:2	
Shape_3|
add_2AddV2MatMul_2:output:0Cast:y:0*
T0*=
_output_shapes+
):'???????????????????????????2
add_2_
Greater_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
Greater_1/y?
	Greater_1Greater	add_2:z:0Greater_1/y:output:0*
T0*=
_output_shapes+
):'???????????????????????????2
	Greater_1~
Cast_1CastGreater_1:z:0*

DstT0*

SrcT0
*=
_output_shapes+
):'???????????????????????????2
Cast_1y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose
Cast_1:y:0transpose_1/perm:output:0*
T0*=
_output_shapes+
):'???????????????????????????2
transpose_1?
MatMul_3BatchMatMulV2transpose_1:y:0	add_1:z:0*
T0*4
_output_shapes"
 :??????????????????2

MatMul_3S
Shape_4ShapeMatMul_3:output:0*
T0*
_output_shapes
:2	
Shape_4p
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indices?
SumSum
Cast_1:y:0Sum/reduction_indices:output:0*
T0*4
_output_shapes"
 :??????????????????*
	keep_dims(2
SumW
add_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32	
add_3/yv
add_3AddV2Sum:output:0add_3/y:output:0*
T0*4
_output_shapes"
 :??????????????????2
add_3z
truedivRealDivMatMul_3:output:0	add_3:z:0*
T0*4
_output_shapes"
 :??????????????????2	
truediv`
ReluRelutruediv:z:0*
T0*4
_output_shapes"
 :??????????????????2
Reluz
IdentityIdentityRelu:activations:0^NoOp*
T0*4
_output_shapes"
 :??????????????????2

Identity?
NoOpNoOp^add_1/ReadVariableOp^transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*`
_input_shapesO
M:?????????????????? :'???????????????????????????: : 2,
add_1/ReadVariableOpadd_1/ReadVariableOp24
transpose/ReadVariableOptranspose/ReadVariableOp:\ X
4
_output_shapes"
 :?????????????????? 
 
_user_specified_nameinputs:ea
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?
?
*__inference_model_12_layer_call_fn_4263051
input_49
input_50
input_51
input_52
unknown:@
	unknown_0:@
	unknown_1:@ 
	unknown_2: 
	unknown_3: 
	unknown_4:
	unknown_5:
	unknown_6: 
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:

unknown_16:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_49input_50input_51input_52unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*!
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_model_12_layer_call_and_return_conditional_losses_42630122
StatefulPartitionedCallr
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????	:'???????????????????????????:?????????	:'???????????????????????????: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:?????????	
"
_user_specified_name
input_49:gc
=
_output_shapes+
):'???????????????????????????
"
_user_specified_name
input_50:UQ
+
_output_shapes
:?????????	
"
_user_specified_name
input_51:gc
=
_output_shapes+
):'???????????????????????????
"
_user_specified_name
input_52
?

?
E__inference_dense_50_layer_call_and_return_conditional_losses_4262988

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOpj
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpx
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2	
BiasAddO
ReluReluBiasAdd:output:0*
T0*
_output_shapes

:2
Relud
IdentityIdentityRelu:activations:0^NoOp*
T0*
_output_shapes

:2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
:: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:F B

_output_shapes

:
 
_user_specified_nameinputs
?-
?
J__inference_graph_conv_36_layer_call_and_return_conditional_losses_4264957
inputs_0
inputs_11
shape_2_readvariableop_resource:@+
add_1_readvariableop_resource:@
identity??add_1/ReadVariableOp?transpose/ReadVariableOp}
MatMulBatchMatMulV2inputs_1inputs_1*
T0*=
_output_shapes+
):'???????????????????????????2
MatMulM
ShapeShapeMatMul:output:0*
T0*
_output_shapes
:2
Shapev
addAddV2MatMul:output:0inputs_1*
T0*=
_output_shapes+
):'???????????????????????????2
add[
	Greater/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
	Greater/y?
GreaterGreateradd:z:0Greater/y:output:0*
T0*=
_output_shapes+
):'???????????????????????????2	
Greaterx
CastCastGreater:z:0*

DstT0*

SrcT0
*=
_output_shapes+
):'???????????????????????????2
CastJ
Shape_1Shapeinputs_0*
T0*
_output_shapes
:2	
Shape_1^
unstackUnpackShape_1:output:0*
T0*
_output_shapes
: : : *	
num2	
unstack?
Shape_2/ReadVariableOpReadVariableOpshape_2_readvariableop_resource*
_output_shapes

:@*
dtype02
Shape_2/ReadVariableOpc
Shape_2Const*
_output_shapes
:*
dtype0*
valueB"   @   2	
Shape_2`
	unstack_1UnpackShape_2:output:0*
T0*
_output_shapes
: : *	
num2
	unstack_1o
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape/shapeq
ReshapeReshapeinputs_0Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2	
Reshape?
transpose/ReadVariableOpReadVariableOpshape_2_readvariableop_resource*
_output_shapes

:@*
dtype02
transpose/ReadVariableOpq
transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose/perm?
	transpose	Transpose transpose/ReadVariableOp:value:0transpose/perm:output:0*
T0*
_output_shapes

:@2
	transposes
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ????2
Reshape_1/shapes
	Reshape_1Reshapetranspose:y:0Reshape_1/shape:output:0*
T0*
_output_shapes

:@2
	Reshape_1v
MatMul_1MatMulReshape:output:0Reshape_1:output:0*
T0*'
_output_shapes
:?????????@2

MatMul_1h
Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :	2
Reshape_2/shape/1h
Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :@2
Reshape_2/shape/2?
Reshape_2/shapePackunstack:output:0Reshape_2/shape/1:output:0Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape_2/shape?
	Reshape_2ReshapeMatMul_1:product:0Reshape_2/shape:output:0*
T0*+
_output_shapes
:?????????	@2
	Reshape_2?
add_1/ReadVariableOpReadVariableOpadd_1_readvariableop_resource*
_output_shapes
:@*
dtype02
add_1/ReadVariableOp
add_1AddV2Reshape_2:output:0add_1/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????	@2
add_1?
MatMul_2BatchMatMulV2Cast:y:0Cast:y:0*
T0*=
_output_shapes+
):'???????????????????????????2

MatMul_2S
Shape_3ShapeMatMul_2:output:0*
T0*
_output_shapes
:2	
Shape_3|
add_2AddV2MatMul_2:output:0Cast:y:0*
T0*=
_output_shapes+
):'???????????????????????????2
add_2_
Greater_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
Greater_1/y?
	Greater_1Greater	add_2:z:0Greater_1/y:output:0*
T0*=
_output_shapes+
):'???????????????????????????2
	Greater_1~
Cast_1CastGreater_1:z:0*

DstT0*

SrcT0
*=
_output_shapes+
):'???????????????????????????2
Cast_1y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose
Cast_1:y:0transpose_1/perm:output:0*
T0*=
_output_shapes+
):'???????????????????????????2
transpose_1?
MatMul_3BatchMatMulV2transpose_1:y:0	add_1:z:0*
T0*4
_output_shapes"
 :??????????????????@2

MatMul_3S
Shape_4ShapeMatMul_3:output:0*
T0*
_output_shapes
:2	
Shape_4p
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indices?
SumSum
Cast_1:y:0Sum/reduction_indices:output:0*
T0*4
_output_shapes"
 :??????????????????*
	keep_dims(2
SumW
add_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32	
add_3/yv
add_3AddV2Sum:output:0add_3/y:output:0*
T0*4
_output_shapes"
 :??????????????????2
add_3z
truedivRealDivMatMul_3:output:0	add_3:z:0*
T0*4
_output_shapes"
 :??????????????????@2	
truediv`
ReluRelutruediv:z:0*
T0*4
_output_shapes"
 :??????????????????@2
Reluz
IdentityIdentityRelu:activations:0^NoOp*
T0*4
_output_shapes"
 :??????????????????@2

Identity?
NoOpNoOp^add_1/ReadVariableOp^transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:?????????	:'???????????????????????????: : 2,
add_1/ReadVariableOpadd_1/ReadVariableOp24
transpose/ReadVariableOptranspose/ReadVariableOp:U Q
+
_output_shapes
:?????????	
"
_user_specified_name
inputs/0:gc
=
_output_shapes+
):'???????????????????????????
"
_user_specified_name
inputs/1
?
?
%__inference_signature_wrapper_4263724
input_49
input_50
input_51
input_52
unknown:@
	unknown_0:@
	unknown_1:@ 
	unknown_2: 
	unknown_3: 
	unknown_4:
	unknown_5:
	unknown_6: 
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:

unknown_13:

unknown_14:

unknown_15:

unknown_16:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_49input_50input_51input_52unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*!
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference__wrapped_model_42625092
StatefulPartitionedCallr
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????	:'???????????????????????????:?????????	:'???????????????????????????: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:?????????	
"
_user_specified_name
input_49:gc
=
_output_shapes+
):'???????????????????????????
"
_user_specified_name
input_50:UQ
+
_output_shapes
:?????????	
"
_user_specified_name
input_51:gc
=
_output_shapes+
):'???????????????????????????
"
_user_specified_name
input_52
?

?
E__inference_dense_48_layer_call_and_return_conditional_losses_4265508

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOpj
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpx
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2	
BiasAddO
ReluReluBiasAdd:output:0*
T0*
_output_shapes

:2
Relud
IdentityIdentityRelu:activations:0^NoOp*
T0*
_output_shapes

:2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
:: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:F B

_output_shapes

:
 
_user_specified_nameinputs
?,
?
J__inference_graph_conv_38_layer_call_and_return_conditional_losses_4265198
inputs_0
inputs_11
shape_2_readvariableop_resource: +
add_1_readvariableop_resource:
identity??add_1/ReadVariableOp?transpose/ReadVariableOp}
MatMulBatchMatMulV2inputs_1inputs_1*
T0*=
_output_shapes+
):'???????????????????????????2
MatMulM
ShapeShapeMatMul:output:0*
T0*
_output_shapes
:2
Shapev
addAddV2MatMul:output:0inputs_1*
T0*=
_output_shapes+
):'???????????????????????????2
add[
	Greater/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
	Greater/y?
GreaterGreateradd:z:0Greater/y:output:0*
T0*=
_output_shapes+
):'???????????????????????????2	
Greaterx
CastCastGreater:z:0*

DstT0*

SrcT0
*=
_output_shapes+
):'???????????????????????????2
CastJ
Shape_1Shapeinputs_0*
T0*
_output_shapes
:2	
Shape_1^
unstackUnpackShape_1:output:0*
T0*
_output_shapes
: : : *	
num2	
unstack?
Shape_2/ReadVariableOpReadVariableOpshape_2_readvariableop_resource*
_output_shapes

: *
dtype02
Shape_2/ReadVariableOpc
Shape_2Const*
_output_shapes
:*
dtype0*
valueB"       2	
Shape_2`
	unstack_1UnpackShape_2:output:0*
T0*
_output_shapes
: : *	
num2
	unstack_1o
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????    2
Reshape/shapeq
ReshapeReshapeinputs_0Reshape/shape:output:0*
T0*'
_output_shapes
:????????? 2	
Reshape?
transpose/ReadVariableOpReadVariableOpshape_2_readvariableop_resource*
_output_shapes

: *
dtype02
transpose/ReadVariableOpq
transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose/perm?
	transpose	Transpose transpose/ReadVariableOp:value:0transpose/perm:output:0*
T0*
_output_shapes

: 2
	transposes
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"    ????2
Reshape_1/shapes
	Reshape_1Reshapetranspose:y:0Reshape_1/shape:output:0*
T0*
_output_shapes

: 2
	Reshape_1v
MatMul_1MatMulReshape:output:0Reshape_1:output:0*
T0*'
_output_shapes
:?????????2

MatMul_1h
Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape_2/shape/2?
Reshape_2/shapePackunstack:output:0unstack:output:1Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape_2/shape?
	Reshape_2ReshapeMatMul_1:product:0Reshape_2/shape:output:0*
T0*4
_output_shapes"
 :??????????????????2
	Reshape_2?
add_1/ReadVariableOpReadVariableOpadd_1_readvariableop_resource*
_output_shapes
:*
dtype02
add_1/ReadVariableOp?
add_1AddV2Reshape_2:output:0add_1/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :??????????????????2
add_1?
MatMul_2BatchMatMulV2Cast:y:0Cast:y:0*
T0*=
_output_shapes+
):'???????????????????????????2

MatMul_2S
Shape_3ShapeMatMul_2:output:0*
T0*
_output_shapes
:2	
Shape_3|
add_2AddV2MatMul_2:output:0Cast:y:0*
T0*=
_output_shapes+
):'???????????????????????????2
add_2_
Greater_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
Greater_1/y?
	Greater_1Greater	add_2:z:0Greater_1/y:output:0*
T0*=
_output_shapes+
):'???????????????????????????2
	Greater_1~
Cast_1CastGreater_1:z:0*

DstT0*

SrcT0
*=
_output_shapes+
):'???????????????????????????2
Cast_1y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose
Cast_1:y:0transpose_1/perm:output:0*
T0*=
_output_shapes+
):'???????????????????????????2
transpose_1?
MatMul_3BatchMatMulV2transpose_1:y:0	add_1:z:0*
T0*4
_output_shapes"
 :??????????????????2

MatMul_3S
Shape_4ShapeMatMul_3:output:0*
T0*
_output_shapes
:2	
Shape_4p
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indices?
SumSum
Cast_1:y:0Sum/reduction_indices:output:0*
T0*4
_output_shapes"
 :??????????????????*
	keep_dims(2
SumW
add_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32	
add_3/yv
add_3AddV2Sum:output:0add_3/y:output:0*
T0*4
_output_shapes"
 :??????????????????2
add_3z
truedivRealDivMatMul_3:output:0	add_3:z:0*
T0*4
_output_shapes"
 :??????????????????2	
truediv`
ReluRelutruediv:z:0*
T0*4
_output_shapes"
 :??????????????????2
Reluz
IdentityIdentityRelu:activations:0^NoOp*
T0*4
_output_shapes"
 :??????????????????2

Identity?
NoOpNoOp^add_1/ReadVariableOp^transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*`
_input_shapesO
M:?????????????????? :'???????????????????????????: : 2,
add_1/ReadVariableOpadd_1/ReadVariableOp24
transpose/ReadVariableOptranspose/ReadVariableOp:^ Z
4
_output_shapes"
 :?????????????????? 
"
_user_specified_name
inputs/0:gc
=
_output_shapes+
):'???????????????????????????
"
_user_specified_name
inputs/1
?	
?
8__inference_neural_tensor_layer_12_layer_call_fn_4265288
inputs_0
inputs_1
unknown: 
	unknown_0:
	unknown_1:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_neural_tensor_layer_12_layer_call_and_return_conditional_losses_42629352
StatefulPartitionedCallr
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
::: : : 22
StatefulPartitionedCallStatefulPartitionedCall:H D

_output_shapes

:
"
_user_specified_name
inputs/0:HD

_output_shapes

:
"
_user_specified_name
inputs/1
??
?
S__inference_neural_tensor_layer_12_layer_call_and_return_conditional_losses_4262935

inputs
inputs_10
matmul_readvariableop_resource: -
readvariableop_resource:)
add_readvariableop_resource:
identity??MatMul/ReadVariableOp?ReadVariableOp?ReadVariableOp_1?ReadVariableOp_10?ReadVariableOp_11?ReadVariableOp_12?ReadVariableOp_13?ReadVariableOp_14?ReadVariableOp_15?ReadVariableOp_2?ReadVariableOp_3?ReadVariableOp_4?ReadVariableOp_5?ReadVariableOp_6?ReadVariableOp_7?ReadVariableOp_8?ReadVariableOp_9?add/ReadVariableOp?add_1/ReadVariableOp?add_10/ReadVariableOp?add_11/ReadVariableOp?add_12/ReadVariableOp?add_13/ReadVariableOp?add_14/ReadVariableOp?add_15/ReadVariableOp?add_2/ReadVariableOp?add_3/ReadVariableOp?add_4/ReadVariableOp?add_5/ReadVariableOp?add_6/ReadVariableOp?add_7/ReadVariableOp?add_8/ReadVariableOp?add_9/ReadVariableOp_
ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axisv
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*
_output_shapes

: 2
concat?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulconcat:output:0MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
MatMul|
ReadVariableOpReadVariableOpreadvariableop_resource*"
_output_shapes
:*
dtype02
ReadVariableOpx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceReadVariableOp:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
strided_slice_1i
MatMul_1MatMulinputsstrided_slice_1:output:0*
T0*
_output_shapes

:2

MatMul_1X
mulMulinputs_1MatMul_1:product:0*
T0*
_output_shapes

:2
mul?
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:*
dtype02
add/ReadVariableOpa
addAddV2mul:z:0add/ReadVariableOp:value:0*
T0*
_output_shapes

:2
addp
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indices_
SumSumadd:z:0Sum/reduction_indices:output:0*
T0*
_output_shapes
:2
Sum?
ReadVariableOp_1ReadVariableOpreadvariableop_resource*"
_output_shapes
:*
dtype02
ReadVariableOp_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceReadVariableOp_1:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
strided_slice_2i
MatMul_2MatMulinputsstrided_slice_2:output:0*
T0*
_output_shapes

:2

MatMul_2\
mul_1Mulinputs_1MatMul_2:product:0*
T0*
_output_shapes

:2
mul_1?
add_1/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:*
dtype02
add_1/ReadVariableOpi
add_1AddV2	mul_1:z:0add_1/ReadVariableOp:value:0*
T0*
_output_shapes

:2
add_1t
Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum_1/reduction_indicesg
Sum_1Sum	add_1:z:0 Sum_1/reduction_indices:output:0*
T0*
_output_shapes
:2
Sum_1?
ReadVariableOp_2ReadVariableOpreadvariableop_resource*"
_output_shapes
:*
dtype02
ReadVariableOp_2x
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSliceReadVariableOp_2:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
strided_slice_3i
MatMul_3MatMulinputsstrided_slice_3:output:0*
T0*
_output_shapes

:2

MatMul_3\
mul_2Mulinputs_1MatMul_3:product:0*
T0*
_output_shapes

:2
mul_2?
add_2/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:*
dtype02
add_2/ReadVariableOpi
add_2AddV2	mul_2:z:0add_2/ReadVariableOp:value:0*
T0*
_output_shapes

:2
add_2t
Sum_2/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum_2/reduction_indicesg
Sum_2Sum	add_2:z:0 Sum_2/reduction_indices:output:0*
T0*
_output_shapes
:2
Sum_2?
ReadVariableOp_3ReadVariableOpreadvariableop_resource*"
_output_shapes
:*
dtype02
ReadVariableOp_3x
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_4/stack|
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_4/stack_1|
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_4/stack_2?
strided_slice_4StridedSliceReadVariableOp_3:value:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
strided_slice_4i
MatMul_4MatMulinputsstrided_slice_4:output:0*
T0*
_output_shapes

:2

MatMul_4\
mul_3Mulinputs_1MatMul_4:product:0*
T0*
_output_shapes

:2
mul_3?
add_3/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:*
dtype02
add_3/ReadVariableOpi
add_3AddV2	mul_3:z:0add_3/ReadVariableOp:value:0*
T0*
_output_shapes

:2
add_3t
Sum_3/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum_3/reduction_indicesg
Sum_3Sum	add_3:z:0 Sum_3/reduction_indices:output:0*
T0*
_output_shapes
:2
Sum_3?
ReadVariableOp_4ReadVariableOpreadvariableop_resource*"
_output_shapes
:*
dtype02
ReadVariableOp_4x
strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_5/stack|
strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_5/stack_1|
strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_5/stack_2?
strided_slice_5StridedSliceReadVariableOp_4:value:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
strided_slice_5i
MatMul_5MatMulinputsstrided_slice_5:output:0*
T0*
_output_shapes

:2

MatMul_5\
mul_4Mulinputs_1MatMul_5:product:0*
T0*
_output_shapes

:2
mul_4?
add_4/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:*
dtype02
add_4/ReadVariableOpi
add_4AddV2	mul_4:z:0add_4/ReadVariableOp:value:0*
T0*
_output_shapes

:2
add_4t
Sum_4/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum_4/reduction_indicesg
Sum_4Sum	add_4:z:0 Sum_4/reduction_indices:output:0*
T0*
_output_shapes
:2
Sum_4?
ReadVariableOp_5ReadVariableOpreadvariableop_resource*"
_output_shapes
:*
dtype02
ReadVariableOp_5x
strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_6/stack|
strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_6/stack_1|
strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_6/stack_2?
strided_slice_6StridedSliceReadVariableOp_5:value:0strided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
strided_slice_6i
MatMul_6MatMulinputsstrided_slice_6:output:0*
T0*
_output_shapes

:2

MatMul_6\
mul_5Mulinputs_1MatMul_6:product:0*
T0*
_output_shapes

:2
mul_5?
add_5/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:*
dtype02
add_5/ReadVariableOpi
add_5AddV2	mul_5:z:0add_5/ReadVariableOp:value:0*
T0*
_output_shapes

:2
add_5t
Sum_5/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum_5/reduction_indicesg
Sum_5Sum	add_5:z:0 Sum_5/reduction_indices:output:0*
T0*
_output_shapes
:2
Sum_5?
ReadVariableOp_6ReadVariableOpreadvariableop_resource*"
_output_shapes
:*
dtype02
ReadVariableOp_6x
strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_7/stack|
strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_7/stack_1|
strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_7/stack_2?
strided_slice_7StridedSliceReadVariableOp_6:value:0strided_slice_7/stack:output:0 strided_slice_7/stack_1:output:0 strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
strided_slice_7i
MatMul_7MatMulinputsstrided_slice_7:output:0*
T0*
_output_shapes

:2

MatMul_7\
mul_6Mulinputs_1MatMul_7:product:0*
T0*
_output_shapes

:2
mul_6?
add_6/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:*
dtype02
add_6/ReadVariableOpi
add_6AddV2	mul_6:z:0add_6/ReadVariableOp:value:0*
T0*
_output_shapes

:2
add_6t
Sum_6/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum_6/reduction_indicesg
Sum_6Sum	add_6:z:0 Sum_6/reduction_indices:output:0*
T0*
_output_shapes
:2
Sum_6?
ReadVariableOp_7ReadVariableOpreadvariableop_resource*"
_output_shapes
:*
dtype02
ReadVariableOp_7x
strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_8/stack|
strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_8/stack_1|
strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_8/stack_2?
strided_slice_8StridedSliceReadVariableOp_7:value:0strided_slice_8/stack:output:0 strided_slice_8/stack_1:output:0 strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
strided_slice_8i
MatMul_8MatMulinputsstrided_slice_8:output:0*
T0*
_output_shapes

:2

MatMul_8\
mul_7Mulinputs_1MatMul_8:product:0*
T0*
_output_shapes

:2
mul_7?
add_7/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:*
dtype02
add_7/ReadVariableOpi
add_7AddV2	mul_7:z:0add_7/ReadVariableOp:value:0*
T0*
_output_shapes

:2
add_7t
Sum_7/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum_7/reduction_indicesg
Sum_7Sum	add_7:z:0 Sum_7/reduction_indices:output:0*
T0*
_output_shapes
:2
Sum_7?
ReadVariableOp_8ReadVariableOpreadvariableop_resource*"
_output_shapes
:*
dtype02
ReadVariableOp_8x
strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_9/stack|
strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:	2
strided_slice_9/stack_1|
strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_9/stack_2?
strided_slice_9StridedSliceReadVariableOp_8:value:0strided_slice_9/stack:output:0 strided_slice_9/stack_1:output:0 strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
strided_slice_9i
MatMul_9MatMulinputsstrided_slice_9:output:0*
T0*
_output_shapes

:2

MatMul_9\
mul_8Mulinputs_1MatMul_9:product:0*
T0*
_output_shapes

:2
mul_8?
add_8/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:*
dtype02
add_8/ReadVariableOpi
add_8AddV2	mul_8:z:0add_8/ReadVariableOp:value:0*
T0*
_output_shapes

:2
add_8t
Sum_8/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum_8/reduction_indicesg
Sum_8Sum	add_8:z:0 Sum_8/reduction_indices:output:0*
T0*
_output_shapes
:2
Sum_8?
ReadVariableOp_9ReadVariableOpreadvariableop_resource*"
_output_shapes
:*
dtype02
ReadVariableOp_9z
strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB:	2
strided_slice_10/stack~
strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
2
strided_slice_10/stack_1~
strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_10/stack_2?
strided_slice_10StridedSliceReadVariableOp_9:value:0strided_slice_10/stack:output:0!strided_slice_10/stack_1:output:0!strided_slice_10/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
strided_slice_10l
	MatMul_10MatMulinputsstrided_slice_10:output:0*
T0*
_output_shapes

:2
	MatMul_10]
mul_9Mulinputs_1MatMul_10:product:0*
T0*
_output_shapes

:2
mul_9?
add_9/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:*
dtype02
add_9/ReadVariableOpi
add_9AddV2	mul_9:z:0add_9/ReadVariableOp:value:0*
T0*
_output_shapes

:2
add_9t
Sum_9/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum_9/reduction_indicesg
Sum_9Sum	add_9:z:0 Sum_9/reduction_indices:output:0*
T0*
_output_shapes
:2
Sum_9?
ReadVariableOp_10ReadVariableOpreadvariableop_resource*"
_output_shapes
:*
dtype02
ReadVariableOp_10z
strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:
2
strided_slice_11/stack~
strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_11/stack_1~
strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_11/stack_2?
strided_slice_11StridedSliceReadVariableOp_10:value:0strided_slice_11/stack:output:0!strided_slice_11/stack_1:output:0!strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
strided_slice_11l
	MatMul_11MatMulinputsstrided_slice_11:output:0*
T0*
_output_shapes

:2
	MatMul_11_
mul_10Mulinputs_1MatMul_11:product:0*
T0*
_output_shapes

:2
mul_10?
add_10/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:*
dtype02
add_10/ReadVariableOpm
add_10AddV2
mul_10:z:0add_10/ReadVariableOp:value:0*
T0*
_output_shapes

:2
add_10v
Sum_10/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum_10/reduction_indicesk
Sum_10Sum
add_10:z:0!Sum_10/reduction_indices:output:0*
T0*
_output_shapes
:2
Sum_10?
ReadVariableOp_11ReadVariableOpreadvariableop_resource*"
_output_shapes
:*
dtype02
ReadVariableOp_11z
strided_slice_12/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_12/stack~
strided_slice_12/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_12/stack_1~
strided_slice_12/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_12/stack_2?
strided_slice_12StridedSliceReadVariableOp_11:value:0strided_slice_12/stack:output:0!strided_slice_12/stack_1:output:0!strided_slice_12/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
strided_slice_12l
	MatMul_12MatMulinputsstrided_slice_12:output:0*
T0*
_output_shapes

:2
	MatMul_12_
mul_11Mulinputs_1MatMul_12:product:0*
T0*
_output_shapes

:2
mul_11?
add_11/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:*
dtype02
add_11/ReadVariableOpm
add_11AddV2
mul_11:z:0add_11/ReadVariableOp:value:0*
T0*
_output_shapes

:2
add_11v
Sum_11/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum_11/reduction_indicesk
Sum_11Sum
add_11:z:0!Sum_11/reduction_indices:output:0*
T0*
_output_shapes
:2
Sum_11?
ReadVariableOp_12ReadVariableOpreadvariableop_resource*"
_output_shapes
:*
dtype02
ReadVariableOp_12z
strided_slice_13/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_13/stack~
strided_slice_13/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_13/stack_1~
strided_slice_13/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_13/stack_2?
strided_slice_13StridedSliceReadVariableOp_12:value:0strided_slice_13/stack:output:0!strided_slice_13/stack_1:output:0!strided_slice_13/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
strided_slice_13l
	MatMul_13MatMulinputsstrided_slice_13:output:0*
T0*
_output_shapes

:2
	MatMul_13_
mul_12Mulinputs_1MatMul_13:product:0*
T0*
_output_shapes

:2
mul_12?
add_12/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:*
dtype02
add_12/ReadVariableOpm
add_12AddV2
mul_12:z:0add_12/ReadVariableOp:value:0*
T0*
_output_shapes

:2
add_12v
Sum_12/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum_12/reduction_indicesk
Sum_12Sum
add_12:z:0!Sum_12/reduction_indices:output:0*
T0*
_output_shapes
:2
Sum_12?
ReadVariableOp_13ReadVariableOpreadvariableop_resource*"
_output_shapes
:*
dtype02
ReadVariableOp_13z
strided_slice_14/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_14/stack~
strided_slice_14/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_14/stack_1~
strided_slice_14/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_14/stack_2?
strided_slice_14StridedSliceReadVariableOp_13:value:0strided_slice_14/stack:output:0!strided_slice_14/stack_1:output:0!strided_slice_14/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
strided_slice_14l
	MatMul_14MatMulinputsstrided_slice_14:output:0*
T0*
_output_shapes

:2
	MatMul_14_
mul_13Mulinputs_1MatMul_14:product:0*
T0*
_output_shapes

:2
mul_13?
add_13/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:*
dtype02
add_13/ReadVariableOpm
add_13AddV2
mul_13:z:0add_13/ReadVariableOp:value:0*
T0*
_output_shapes

:2
add_13v
Sum_13/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum_13/reduction_indicesk
Sum_13Sum
add_13:z:0!Sum_13/reduction_indices:output:0*
T0*
_output_shapes
:2
Sum_13?
ReadVariableOp_14ReadVariableOpreadvariableop_resource*"
_output_shapes
:*
dtype02
ReadVariableOp_14z
strided_slice_15/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_15/stack~
strided_slice_15/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_15/stack_1~
strided_slice_15/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_15/stack_2?
strided_slice_15StridedSliceReadVariableOp_14:value:0strided_slice_15/stack:output:0!strided_slice_15/stack_1:output:0!strided_slice_15/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
strided_slice_15l
	MatMul_15MatMulinputsstrided_slice_15:output:0*
T0*
_output_shapes

:2
	MatMul_15_
mul_14Mulinputs_1MatMul_15:product:0*
T0*
_output_shapes

:2
mul_14?
add_14/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:*
dtype02
add_14/ReadVariableOpm
add_14AddV2
mul_14:z:0add_14/ReadVariableOp:value:0*
T0*
_output_shapes

:2
add_14v
Sum_14/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum_14/reduction_indicesk
Sum_14Sum
add_14:z:0!Sum_14/reduction_indices:output:0*
T0*
_output_shapes
:2
Sum_14?
ReadVariableOp_15ReadVariableOpreadvariableop_resource*"
_output_shapes
:*
dtype02
ReadVariableOp_15z
strided_slice_16/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_16/stack~
strided_slice_16/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_16/stack_1~
strided_slice_16/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_16/stack_2?
strided_slice_16StridedSliceReadVariableOp_15:value:0strided_slice_16/stack:output:0!strided_slice_16/stack_1:output:0!strided_slice_16/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
strided_slice_16l
	MatMul_16MatMulinputsstrided_slice_16:output:0*
T0*
_output_shapes

:2
	MatMul_16_
mul_15Mulinputs_1MatMul_16:product:0*
T0*
_output_shapes

:2
mul_15?
add_15/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:*
dtype02
add_15/ReadVariableOpm
add_15AddV2
mul_15:z:0add_15/ReadVariableOp:value:0*
T0*
_output_shapes

:2
add_15v
Sum_15/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum_15/reduction_indicesk
Sum_15Sum
add_15:z:0!Sum_15/reduction_indices:output:0*
T0*
_output_shapes
:2
Sum_15`
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
concat_1/axis?
concat_1ConcatV2Sum:output:0Sum_1:output:0Sum_2:output:0Sum_3:output:0Sum_4:output:0Sum_5:output:0Sum_6:output:0Sum_7:output:0Sum_8:output:0Sum_9:output:0Sum_10:output:0Sum_11:output:0Sum_12:output:0Sum_13:output:0Sum_14:output:0Sum_15:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes
:2

concat_1d
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapeq
ReshapeReshapeconcat_1:output:0Reshape/shape:output:0*
T0*
_output_shapes

:2	
Reshapef
add_16AddV2Reshape:output:0MatMul:product:0*
T0*
_output_shapes

:2
add_16I
TanhTanh
add_16:z:0*
T0*
_output_shapes

:2
TanhZ
IdentityIdentityTanh:y:0^NoOp*
T0*
_output_shapes

:2

Identity?
NoOpNoOp^MatMul/ReadVariableOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_10^ReadVariableOp_11^ReadVariableOp_12^ReadVariableOp_13^ReadVariableOp_14^ReadVariableOp_15^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6^ReadVariableOp_7^ReadVariableOp_8^ReadVariableOp_9^add/ReadVariableOp^add_1/ReadVariableOp^add_10/ReadVariableOp^add_11/ReadVariableOp^add_12/ReadVariableOp^add_13/ReadVariableOp^add_14/ReadVariableOp^add_15/ReadVariableOp^add_2/ReadVariableOp^add_3/ReadVariableOp^add_4/ReadVariableOp^add_5/ReadVariableOp^add_6/ReadVariableOp^add_7/ReadVariableOp^add_8/ReadVariableOp^add_9/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
::: : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12&
ReadVariableOp_10ReadVariableOp_102&
ReadVariableOp_11ReadVariableOp_112&
ReadVariableOp_12ReadVariableOp_122&
ReadVariableOp_13ReadVariableOp_132&
ReadVariableOp_14ReadVariableOp_142&
ReadVariableOp_15ReadVariableOp_152$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32$
ReadVariableOp_4ReadVariableOp_42$
ReadVariableOp_5ReadVariableOp_52$
ReadVariableOp_6ReadVariableOp_62$
ReadVariableOp_7ReadVariableOp_72$
ReadVariableOp_8ReadVariableOp_82$
ReadVariableOp_9ReadVariableOp_92(
add/ReadVariableOpadd/ReadVariableOp2,
add_1/ReadVariableOpadd_1/ReadVariableOp2.
add_10/ReadVariableOpadd_10/ReadVariableOp2.
add_11/ReadVariableOpadd_11/ReadVariableOp2.
add_12/ReadVariableOpadd_12/ReadVariableOp2.
add_13/ReadVariableOpadd_13/ReadVariableOp2.
add_14/ReadVariableOpadd_14/ReadVariableOp2.
add_15/ReadVariableOpadd_15/ReadVariableOp2,
add_2/ReadVariableOpadd_2/ReadVariableOp2,
add_3/ReadVariableOpadd_3/ReadVariableOp2,
add_4/ReadVariableOpadd_4/ReadVariableOp2,
add_5/ReadVariableOpadd_5/ReadVariableOp2,
add_6/ReadVariableOpadd_6/ReadVariableOp2,
add_7/ReadVariableOpadd_7/ReadVariableOp2,
add_8/ReadVariableOpadd_8/ReadVariableOp2,
add_9/ReadVariableOpadd_9/ReadVariableOp:F B

_output_shapes

:
 
_user_specified_nameinputs:FB

_output_shapes

:
 
_user_specified_nameinputs
?
?
*__inference_dense_48_layer_call_fn_4265497

inputs
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_48_layer_call_and_return_conditional_losses_42629542
StatefulPartitionedCallr
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
:: : 22
StatefulPartitionedCallStatefulPartitionedCall:F B

_output_shapes

:
 
_user_specified_nameinputs
?
?
*__inference_dense_49_layer_call_fn_4265517

inputs
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_49_layer_call_and_return_conditional_losses_42629712
StatefulPartitionedCallr
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes

:2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
:: : 22
StatefulPartitionedCallStatefulPartitionedCall:F B

_output_shapes

:
 
_user_specified_nameinputs
?]
?

E__inference_model_12_layer_call_and_return_conditional_losses_4263447

inputs
inputs_1
inputs_2
inputs_3'
graph_conv_36_4263381:@#
graph_conv_36_4263383:@'
graph_conv_37_4263389:@ #
graph_conv_37_4263391: '
graph_conv_38_4263397: #
graph_conv_38_4263399:&
attention_12_4263413:0
neural_tensor_layer_12_4263418: 4
neural_tensor_layer_12_4263420:,
neural_tensor_layer_12_4263422:"
dense_48_4263425:
dense_48_4263427:"
dense_49_4263430:
dense_49_4263432:"
dense_50_4263435:
dense_50_4263437:"
dense_51_4263440:
dense_51_4263442:
identity??$attention_12/StatefulPartitionedCall?&attention_12/StatefulPartitionedCall_1? dense_48/StatefulPartitionedCall? dense_49/StatefulPartitionedCall? dense_50/StatefulPartitionedCall? dense_51/StatefulPartitionedCall?%graph_conv_36/StatefulPartitionedCall?'graph_conv_36/StatefulPartitionedCall_1?%graph_conv_37/StatefulPartitionedCall?'graph_conv_37/StatefulPartitionedCall_1?%graph_conv_38/StatefulPartitionedCall?'graph_conv_38/StatefulPartitionedCall_1?.neural_tensor_layer_12/StatefulPartitionedCall?
%graph_conv_36/StatefulPartitionedCallStatefulPartitionedCallinputs_2inputs_3graph_conv_36_4263381graph_conv_36_4263383*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_graph_conv_36_layer_call_and_return_conditional_losses_42633192'
%graph_conv_36/StatefulPartitionedCall?
'graph_conv_36/StatefulPartitionedCall_1StatefulPartitionedCallinputsinputs_1graph_conv_36_4263381graph_conv_36_4263383*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_graph_conv_36_layer_call_and_return_conditional_losses_42633192)
'graph_conv_36/StatefulPartitionedCall_1?
%graph_conv_37/StatefulPartitionedCallStatefulPartitionedCall.graph_conv_36/StatefulPartitionedCall:output:0inputs_3graph_conv_37_4263389graph_conv_37_4263391*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :?????????????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_graph_conv_37_layer_call_and_return_conditional_losses_42632472'
%graph_conv_37/StatefulPartitionedCall?
'graph_conv_37/StatefulPartitionedCall_1StatefulPartitionedCall0graph_conv_36/StatefulPartitionedCall_1:output:0inputs_1graph_conv_37_4263389graph_conv_37_4263391*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :?????????????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_graph_conv_37_layer_call_and_return_conditional_losses_42632472)
'graph_conv_37/StatefulPartitionedCall_1?
%graph_conv_38/StatefulPartitionedCallStatefulPartitionedCall.graph_conv_37/StatefulPartitionedCall:output:0inputs_3graph_conv_38_4263397graph_conv_38_4263399*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_graph_conv_38_layer_call_and_return_conditional_losses_42631762'
%graph_conv_38/StatefulPartitionedCall?
'graph_conv_38/StatefulPartitionedCall_1StatefulPartitionedCall0graph_conv_37/StatefulPartitionedCall_1:output:0inputs_1graph_conv_38_4263397graph_conv_38_4263399*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_graph_conv_38_layer_call_and_return_conditional_losses_42631762)
'graph_conv_38/StatefulPartitionedCall_1?
/tf.__operators__.getitem_25/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/tf.__operators__.getitem_25/strided_slice/stack?
1tf.__operators__.getitem_25/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1tf.__operators__.getitem_25/strided_slice/stack_1?
1tf.__operators__.getitem_25/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1tf.__operators__.getitem_25/strided_slice/stack_2?
)tf.__operators__.getitem_25/strided_sliceStridedSlice.graph_conv_38/StatefulPartitionedCall:output:08tf.__operators__.getitem_25/strided_slice/stack:output:0:tf.__operators__.getitem_25/strided_slice/stack_1:output:0:tf.__operators__.getitem_25/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2+
)tf.__operators__.getitem_25/strided_slice?
/tf.__operators__.getitem_24/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/tf.__operators__.getitem_24/strided_slice/stack?
1tf.__operators__.getitem_24/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1tf.__operators__.getitem_24/strided_slice/stack_1?
1tf.__operators__.getitem_24/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1tf.__operators__.getitem_24/strided_slice/stack_2?
)tf.__operators__.getitem_24/strided_sliceStridedSlice0graph_conv_38/StatefulPartitionedCall_1:output:08tf.__operators__.getitem_24/strided_slice/stack:output:0:tf.__operators__.getitem_24/strided_slice/stack_1:output:0:tf.__operators__.getitem_24/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2+
)tf.__operators__.getitem_24/strided_slice?
$attention_12/StatefulPartitionedCallStatefulPartitionedCall2tf.__operators__.getitem_24/strided_slice:output:0attention_12_4263413*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_attention_12_layer_call_and_return_conditional_losses_42627292&
$attention_12/StatefulPartitionedCall?
&attention_12/StatefulPartitionedCall_1StatefulPartitionedCall2tf.__operators__.getitem_25/strided_slice:output:0attention_12_4263413*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_attention_12_layer_call_and_return_conditional_losses_42627292(
&attention_12/StatefulPartitionedCall_1?
.neural_tensor_layer_12/StatefulPartitionedCallStatefulPartitionedCall-attention_12/StatefulPartitionedCall:output:0/attention_12/StatefulPartitionedCall_1:output:0neural_tensor_layer_12_4263418neural_tensor_layer_12_4263420neural_tensor_layer_12_4263422*
Tin	
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_neural_tensor_layer_12_layer_call_and_return_conditional_losses_426293520
.neural_tensor_layer_12/StatefulPartitionedCall?
 dense_48/StatefulPartitionedCallStatefulPartitionedCall7neural_tensor_layer_12/StatefulPartitionedCall:output:0dense_48_4263425dense_48_4263427*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_48_layer_call_and_return_conditional_losses_42629542"
 dense_48/StatefulPartitionedCall?
 dense_49/StatefulPartitionedCallStatefulPartitionedCall)dense_48/StatefulPartitionedCall:output:0dense_49_4263430dense_49_4263432*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_49_layer_call_and_return_conditional_losses_42629712"
 dense_49/StatefulPartitionedCall?
 dense_50/StatefulPartitionedCallStatefulPartitionedCall)dense_49/StatefulPartitionedCall:output:0dense_50_4263435dense_50_4263437*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_50_layer_call_and_return_conditional_losses_42629882"
 dense_50/StatefulPartitionedCall?
 dense_51/StatefulPartitionedCallStatefulPartitionedCall)dense_50/StatefulPartitionedCall:output:0dense_51_4263440dense_51_4263442*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_51_layer_call_and_return_conditional_losses_42630042"
 dense_51/StatefulPartitionedCall?
tf.math.sigmoid_12/SigmoidSigmoid)dense_51/StatefulPartitionedCall:output:0*
T0*
_output_shapes

:2
tf.math.sigmoid_12/Sigmoidp
IdentityIdentitytf.math.sigmoid_12/Sigmoid:y:0^NoOp*
T0*
_output_shapes

:2

Identity?
NoOpNoOp%^attention_12/StatefulPartitionedCall'^attention_12/StatefulPartitionedCall_1!^dense_48/StatefulPartitionedCall!^dense_49/StatefulPartitionedCall!^dense_50/StatefulPartitionedCall!^dense_51/StatefulPartitionedCall&^graph_conv_36/StatefulPartitionedCall(^graph_conv_36/StatefulPartitionedCall_1&^graph_conv_37/StatefulPartitionedCall(^graph_conv_37/StatefulPartitionedCall_1&^graph_conv_38/StatefulPartitionedCall(^graph_conv_38/StatefulPartitionedCall_1/^neural_tensor_layer_12/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????	:'???????????????????????????:?????????	:'???????????????????????????: : : : : : : : : : : : : : : : : : 2L
$attention_12/StatefulPartitionedCall$attention_12/StatefulPartitionedCall2P
&attention_12/StatefulPartitionedCall_1&attention_12/StatefulPartitionedCall_12D
 dense_48/StatefulPartitionedCall dense_48/StatefulPartitionedCall2D
 dense_49/StatefulPartitionedCall dense_49/StatefulPartitionedCall2D
 dense_50/StatefulPartitionedCall dense_50/StatefulPartitionedCall2D
 dense_51/StatefulPartitionedCall dense_51/StatefulPartitionedCall2N
%graph_conv_36/StatefulPartitionedCall%graph_conv_36/StatefulPartitionedCall2R
'graph_conv_36/StatefulPartitionedCall_1'graph_conv_36/StatefulPartitionedCall_12N
%graph_conv_37/StatefulPartitionedCall%graph_conv_37/StatefulPartitionedCall2R
'graph_conv_37/StatefulPartitionedCall_1'graph_conv_37/StatefulPartitionedCall_12N
%graph_conv_38/StatefulPartitionedCall%graph_conv_38/StatefulPartitionedCall2R
'graph_conv_38/StatefulPartitionedCall_1'graph_conv_38/StatefulPartitionedCall_12`
.neural_tensor_layer_12/StatefulPartitionedCall.neural_tensor_layer_12/StatefulPartitionedCall:S O
+
_output_shapes
:?????????	
 
_user_specified_nameinputs:ea
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs:SO
+
_output_shapes
:?????????	
 
_user_specified_nameinputs:ea
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?-
?
J__inference_graph_conv_36_layer_call_and_return_conditional_losses_4265008
inputs_0
inputs_11
shape_2_readvariableop_resource:@+
add_1_readvariableop_resource:@
identity??add_1/ReadVariableOp?transpose/ReadVariableOp}
MatMulBatchMatMulV2inputs_1inputs_1*
T0*=
_output_shapes+
):'???????????????????????????2
MatMulM
ShapeShapeMatMul:output:0*
T0*
_output_shapes
:2
Shapev
addAddV2MatMul:output:0inputs_1*
T0*=
_output_shapes+
):'???????????????????????????2
add[
	Greater/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
	Greater/y?
GreaterGreateradd:z:0Greater/y:output:0*
T0*=
_output_shapes+
):'???????????????????????????2	
Greaterx
CastCastGreater:z:0*

DstT0*

SrcT0
*=
_output_shapes+
):'???????????????????????????2
CastJ
Shape_1Shapeinputs_0*
T0*
_output_shapes
:2	
Shape_1^
unstackUnpackShape_1:output:0*
T0*
_output_shapes
: : : *	
num2	
unstack?
Shape_2/ReadVariableOpReadVariableOpshape_2_readvariableop_resource*
_output_shapes

:@*
dtype02
Shape_2/ReadVariableOpc
Shape_2Const*
_output_shapes
:*
dtype0*
valueB"   @   2	
Shape_2`
	unstack_1UnpackShape_2:output:0*
T0*
_output_shapes
: : *	
num2
	unstack_1o
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape/shapeq
ReshapeReshapeinputs_0Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2	
Reshape?
transpose/ReadVariableOpReadVariableOpshape_2_readvariableop_resource*
_output_shapes

:@*
dtype02
transpose/ReadVariableOpq
transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose/perm?
	transpose	Transpose transpose/ReadVariableOp:value:0transpose/perm:output:0*
T0*
_output_shapes

:@2
	transposes
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ????2
Reshape_1/shapes
	Reshape_1Reshapetranspose:y:0Reshape_1/shape:output:0*
T0*
_output_shapes

:@2
	Reshape_1v
MatMul_1MatMulReshape:output:0Reshape_1:output:0*
T0*'
_output_shapes
:?????????@2

MatMul_1h
Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :	2
Reshape_2/shape/1h
Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :@2
Reshape_2/shape/2?
Reshape_2/shapePackunstack:output:0Reshape_2/shape/1:output:0Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape_2/shape?
	Reshape_2ReshapeMatMul_1:product:0Reshape_2/shape:output:0*
T0*+
_output_shapes
:?????????	@2
	Reshape_2?
add_1/ReadVariableOpReadVariableOpadd_1_readvariableop_resource*
_output_shapes
:@*
dtype02
add_1/ReadVariableOp
add_1AddV2Reshape_2:output:0add_1/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????	@2
add_1?
MatMul_2BatchMatMulV2Cast:y:0Cast:y:0*
T0*=
_output_shapes+
):'???????????????????????????2

MatMul_2S
Shape_3ShapeMatMul_2:output:0*
T0*
_output_shapes
:2	
Shape_3|
add_2AddV2MatMul_2:output:0Cast:y:0*
T0*=
_output_shapes+
):'???????????????????????????2
add_2_
Greater_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
Greater_1/y?
	Greater_1Greater	add_2:z:0Greater_1/y:output:0*
T0*=
_output_shapes+
):'???????????????????????????2
	Greater_1~
Cast_1CastGreater_1:z:0*

DstT0*

SrcT0
*=
_output_shapes+
):'???????????????????????????2
Cast_1y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose
Cast_1:y:0transpose_1/perm:output:0*
T0*=
_output_shapes+
):'???????????????????????????2
transpose_1?
MatMul_3BatchMatMulV2transpose_1:y:0	add_1:z:0*
T0*4
_output_shapes"
 :??????????????????@2

MatMul_3S
Shape_4ShapeMatMul_3:output:0*
T0*
_output_shapes
:2	
Shape_4p
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indices?
SumSum
Cast_1:y:0Sum/reduction_indices:output:0*
T0*4
_output_shapes"
 :??????????????????*
	keep_dims(2
SumW
add_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32	
add_3/yv
add_3AddV2Sum:output:0add_3/y:output:0*
T0*4
_output_shapes"
 :??????????????????2
add_3z
truedivRealDivMatMul_3:output:0	add_3:z:0*
T0*4
_output_shapes"
 :??????????????????@2	
truediv`
ReluRelutruediv:z:0*
T0*4
_output_shapes"
 :??????????????????@2
Reluz
IdentityIdentityRelu:activations:0^NoOp*
T0*4
_output_shapes"
 :??????????????????@2

Identity?
NoOpNoOp^add_1/ReadVariableOp^transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:?????????	:'???????????????????????????: : 2,
add_1/ReadVariableOpadd_1/ReadVariableOp24
transpose/ReadVariableOptranspose/ReadVariableOp:U Q
+
_output_shapes
:?????????	
"
_user_specified_name
inputs/0:gc
=
_output_shapes+
):'???????????????????????????
"
_user_specified_name
inputs/1
?,
?
J__inference_graph_conv_38_layer_call_and_return_conditional_losses_4265248
inputs_0
inputs_11
shape_2_readvariableop_resource: +
add_1_readvariableop_resource:
identity??add_1/ReadVariableOp?transpose/ReadVariableOp}
MatMulBatchMatMulV2inputs_1inputs_1*
T0*=
_output_shapes+
):'???????????????????????????2
MatMulM
ShapeShapeMatMul:output:0*
T0*
_output_shapes
:2
Shapev
addAddV2MatMul:output:0inputs_1*
T0*=
_output_shapes+
):'???????????????????????????2
add[
	Greater/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
	Greater/y?
GreaterGreateradd:z:0Greater/y:output:0*
T0*=
_output_shapes+
):'???????????????????????????2	
Greaterx
CastCastGreater:z:0*

DstT0*

SrcT0
*=
_output_shapes+
):'???????????????????????????2
CastJ
Shape_1Shapeinputs_0*
T0*
_output_shapes
:2	
Shape_1^
unstackUnpackShape_1:output:0*
T0*
_output_shapes
: : : *	
num2	
unstack?
Shape_2/ReadVariableOpReadVariableOpshape_2_readvariableop_resource*
_output_shapes

: *
dtype02
Shape_2/ReadVariableOpc
Shape_2Const*
_output_shapes
:*
dtype0*
valueB"       2	
Shape_2`
	unstack_1UnpackShape_2:output:0*
T0*
_output_shapes
: : *	
num2
	unstack_1o
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????    2
Reshape/shapeq
ReshapeReshapeinputs_0Reshape/shape:output:0*
T0*'
_output_shapes
:????????? 2	
Reshape?
transpose/ReadVariableOpReadVariableOpshape_2_readvariableop_resource*
_output_shapes

: *
dtype02
transpose/ReadVariableOpq
transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose/perm?
	transpose	Transpose transpose/ReadVariableOp:value:0transpose/perm:output:0*
T0*
_output_shapes

: 2
	transposes
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"    ????2
Reshape_1/shapes
	Reshape_1Reshapetranspose:y:0Reshape_1/shape:output:0*
T0*
_output_shapes

: 2
	Reshape_1v
MatMul_1MatMulReshape:output:0Reshape_1:output:0*
T0*'
_output_shapes
:?????????2

MatMul_1h
Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape_2/shape/2?
Reshape_2/shapePackunstack:output:0unstack:output:1Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape_2/shape?
	Reshape_2ReshapeMatMul_1:product:0Reshape_2/shape:output:0*
T0*4
_output_shapes"
 :??????????????????2
	Reshape_2?
add_1/ReadVariableOpReadVariableOpadd_1_readvariableop_resource*
_output_shapes
:*
dtype02
add_1/ReadVariableOp?
add_1AddV2Reshape_2:output:0add_1/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :??????????????????2
add_1?
MatMul_2BatchMatMulV2Cast:y:0Cast:y:0*
T0*=
_output_shapes+
):'???????????????????????????2

MatMul_2S
Shape_3ShapeMatMul_2:output:0*
T0*
_output_shapes
:2	
Shape_3|
add_2AddV2MatMul_2:output:0Cast:y:0*
T0*=
_output_shapes+
):'???????????????????????????2
add_2_
Greater_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
Greater_1/y?
	Greater_1Greater	add_2:z:0Greater_1/y:output:0*
T0*=
_output_shapes+
):'???????????????????????????2
	Greater_1~
Cast_1CastGreater_1:z:0*

DstT0*

SrcT0
*=
_output_shapes+
):'???????????????????????????2
Cast_1y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose
Cast_1:y:0transpose_1/perm:output:0*
T0*=
_output_shapes+
):'???????????????????????????2
transpose_1?
MatMul_3BatchMatMulV2transpose_1:y:0	add_1:z:0*
T0*4
_output_shapes"
 :??????????????????2

MatMul_3S
Shape_4ShapeMatMul_3:output:0*
T0*
_output_shapes
:2	
Shape_4p
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indices?
SumSum
Cast_1:y:0Sum/reduction_indices:output:0*
T0*4
_output_shapes"
 :??????????????????*
	keep_dims(2
SumW
add_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32	
add_3/yv
add_3AddV2Sum:output:0add_3/y:output:0*
T0*4
_output_shapes"
 :??????????????????2
add_3z
truedivRealDivMatMul_3:output:0	add_3:z:0*
T0*4
_output_shapes"
 :??????????????????2	
truediv`
ReluRelutruediv:z:0*
T0*4
_output_shapes"
 :??????????????????2
Reluz
IdentityIdentityRelu:activations:0^NoOp*
T0*4
_output_shapes"
 :??????????????????2

Identity?
NoOpNoOp^add_1/ReadVariableOp^transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*`
_input_shapesO
M:?????????????????? :'???????????????????????????: : 2,
add_1/ReadVariableOpadd_1/ReadVariableOp24
transpose/ReadVariableOptranspose/ReadVariableOp:^ Z
4
_output_shapes"
 :?????????????????? 
"
_user_specified_name
inputs/0:gc
=
_output_shapes+
):'???????????????????????????
"
_user_specified_name
inputs/1
?-
?
J__inference_graph_conv_36_layer_call_and_return_conditional_losses_4262573

inputs
inputs_11
shape_2_readvariableop_resource:@+
add_1_readvariableop_resource:@
identity??add_1/ReadVariableOp?transpose/ReadVariableOp}
MatMulBatchMatMulV2inputs_1inputs_1*
T0*=
_output_shapes+
):'???????????????????????????2
MatMulM
ShapeShapeMatMul:output:0*
T0*
_output_shapes
:2
Shapev
addAddV2MatMul:output:0inputs_1*
T0*=
_output_shapes+
):'???????????????????????????2
add[
	Greater/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
	Greater/y?
GreaterGreateradd:z:0Greater/y:output:0*
T0*=
_output_shapes+
):'???????????????????????????2	
Greaterx
CastCastGreater:z:0*

DstT0*

SrcT0
*=
_output_shapes+
):'???????????????????????????2
CastH
Shape_1Shapeinputs*
T0*
_output_shapes
:2	
Shape_1^
unstackUnpackShape_1:output:0*
T0*
_output_shapes
: : : *	
num2	
unstack?
Shape_2/ReadVariableOpReadVariableOpshape_2_readvariableop_resource*
_output_shapes

:@*
dtype02
Shape_2/ReadVariableOpc
Shape_2Const*
_output_shapes
:*
dtype0*
valueB"   @   2	
Shape_2`
	unstack_1UnpackShape_2:output:0*
T0*
_output_shapes
: : *	
num2
	unstack_1o
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
Reshape/shapeo
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:?????????2	
Reshape?
transpose/ReadVariableOpReadVariableOpshape_2_readvariableop_resource*
_output_shapes

:@*
dtype02
transpose/ReadVariableOpq
transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose/perm?
	transpose	Transpose transpose/ReadVariableOp:value:0transpose/perm:output:0*
T0*
_output_shapes

:@2
	transposes
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ????2
Reshape_1/shapes
	Reshape_1Reshapetranspose:y:0Reshape_1/shape:output:0*
T0*
_output_shapes

:@2
	Reshape_1v
MatMul_1MatMulReshape:output:0Reshape_1:output:0*
T0*'
_output_shapes
:?????????@2

MatMul_1h
Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :	2
Reshape_2/shape/1h
Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :@2
Reshape_2/shape/2?
Reshape_2/shapePackunstack:output:0Reshape_2/shape/1:output:0Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape_2/shape?
	Reshape_2ReshapeMatMul_1:product:0Reshape_2/shape:output:0*
T0*+
_output_shapes
:?????????	@2
	Reshape_2?
add_1/ReadVariableOpReadVariableOpadd_1_readvariableop_resource*
_output_shapes
:@*
dtype02
add_1/ReadVariableOp
add_1AddV2Reshape_2:output:0add_1/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????	@2
add_1?
MatMul_2BatchMatMulV2Cast:y:0Cast:y:0*
T0*=
_output_shapes+
):'???????????????????????????2

MatMul_2S
Shape_3ShapeMatMul_2:output:0*
T0*
_output_shapes
:2	
Shape_3|
add_2AddV2MatMul_2:output:0Cast:y:0*
T0*=
_output_shapes+
):'???????????????????????????2
add_2_
Greater_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
Greater_1/y?
	Greater_1Greater	add_2:z:0Greater_1/y:output:0*
T0*=
_output_shapes+
):'???????????????????????????2
	Greater_1~
Cast_1CastGreater_1:z:0*

DstT0*

SrcT0
*=
_output_shapes+
):'???????????????????????????2
Cast_1y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose
Cast_1:y:0transpose_1/perm:output:0*
T0*=
_output_shapes+
):'???????????????????????????2
transpose_1?
MatMul_3BatchMatMulV2transpose_1:y:0	add_1:z:0*
T0*4
_output_shapes"
 :??????????????????@2

MatMul_3S
Shape_4ShapeMatMul_3:output:0*
T0*
_output_shapes
:2	
Shape_4p
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indices?
SumSum
Cast_1:y:0Sum/reduction_indices:output:0*
T0*4
_output_shapes"
 :??????????????????*
	keep_dims(2
SumW
add_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32	
add_3/yv
add_3AddV2Sum:output:0add_3/y:output:0*
T0*4
_output_shapes"
 :??????????????????2
add_3z
truedivRealDivMatMul_3:output:0	add_3:z:0*
T0*4
_output_shapes"
 :??????????????????@2	
truediv`
ReluRelutruediv:z:0*
T0*4
_output_shapes"
 :??????????????????@2
Reluz
IdentityIdentityRelu:activations:0^NoOp*
T0*4
_output_shapes"
 :??????????????????@2

Identity?
NoOpNoOp^add_1/ReadVariableOp^transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:?????????	:'???????????????????????????: : 2,
add_1/ReadVariableOpadd_1/ReadVariableOp24
transpose/ReadVariableOptranspose/ReadVariableOp:S O
+
_output_shapes
:?????????	
 
_user_specified_nameinputs:ea
=
_output_shapes+
):'???????????????????????????
 
_user_specified_nameinputs
?]
?

E__inference_model_12_layer_call_and_return_conditional_losses_4263674
input_49
input_50
input_51
input_52'
graph_conv_36_4263608:@#
graph_conv_36_4263610:@'
graph_conv_37_4263616:@ #
graph_conv_37_4263618: '
graph_conv_38_4263624: #
graph_conv_38_4263626:&
attention_12_4263640:0
neural_tensor_layer_12_4263645: 4
neural_tensor_layer_12_4263647:,
neural_tensor_layer_12_4263649:"
dense_48_4263652:
dense_48_4263654:"
dense_49_4263657:
dense_49_4263659:"
dense_50_4263662:
dense_50_4263664:"
dense_51_4263667:
dense_51_4263669:
identity??$attention_12/StatefulPartitionedCall?&attention_12/StatefulPartitionedCall_1? dense_48/StatefulPartitionedCall? dense_49/StatefulPartitionedCall? dense_50/StatefulPartitionedCall? dense_51/StatefulPartitionedCall?%graph_conv_36/StatefulPartitionedCall?'graph_conv_36/StatefulPartitionedCall_1?%graph_conv_37/StatefulPartitionedCall?'graph_conv_37/StatefulPartitionedCall_1?%graph_conv_38/StatefulPartitionedCall?'graph_conv_38/StatefulPartitionedCall_1?.neural_tensor_layer_12/StatefulPartitionedCall?
%graph_conv_36/StatefulPartitionedCallStatefulPartitionedCallinput_51input_52graph_conv_36_4263608graph_conv_36_4263610*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_graph_conv_36_layer_call_and_return_conditional_losses_42633192'
%graph_conv_36/StatefulPartitionedCall?
'graph_conv_36/StatefulPartitionedCall_1StatefulPartitionedCallinput_49input_50graph_conv_36_4263608graph_conv_36_4263610*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_graph_conv_36_layer_call_and_return_conditional_losses_42633192)
'graph_conv_36/StatefulPartitionedCall_1?
%graph_conv_37/StatefulPartitionedCallStatefulPartitionedCall.graph_conv_36/StatefulPartitionedCall:output:0input_52graph_conv_37_4263616graph_conv_37_4263618*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :?????????????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_graph_conv_37_layer_call_and_return_conditional_losses_42632472'
%graph_conv_37/StatefulPartitionedCall?
'graph_conv_37/StatefulPartitionedCall_1StatefulPartitionedCall0graph_conv_36/StatefulPartitionedCall_1:output:0input_50graph_conv_37_4263616graph_conv_37_4263618*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :?????????????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_graph_conv_37_layer_call_and_return_conditional_losses_42632472)
'graph_conv_37/StatefulPartitionedCall_1?
%graph_conv_38/StatefulPartitionedCallStatefulPartitionedCall.graph_conv_37/StatefulPartitionedCall:output:0input_52graph_conv_38_4263624graph_conv_38_4263626*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_graph_conv_38_layer_call_and_return_conditional_losses_42631762'
%graph_conv_38/StatefulPartitionedCall?
'graph_conv_38/StatefulPartitionedCall_1StatefulPartitionedCall0graph_conv_37/StatefulPartitionedCall_1:output:0input_50graph_conv_38_4263624graph_conv_38_4263626*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_graph_conv_38_layer_call_and_return_conditional_losses_42631762)
'graph_conv_38/StatefulPartitionedCall_1?
/tf.__operators__.getitem_25/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/tf.__operators__.getitem_25/strided_slice/stack?
1tf.__operators__.getitem_25/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1tf.__operators__.getitem_25/strided_slice/stack_1?
1tf.__operators__.getitem_25/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1tf.__operators__.getitem_25/strided_slice/stack_2?
)tf.__operators__.getitem_25/strided_sliceStridedSlice.graph_conv_38/StatefulPartitionedCall:output:08tf.__operators__.getitem_25/strided_slice/stack:output:0:tf.__operators__.getitem_25/strided_slice/stack_1:output:0:tf.__operators__.getitem_25/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2+
)tf.__operators__.getitem_25/strided_slice?
/tf.__operators__.getitem_24/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/tf.__operators__.getitem_24/strided_slice/stack?
1tf.__operators__.getitem_24/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1tf.__operators__.getitem_24/strided_slice/stack_1?
1tf.__operators__.getitem_24/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1tf.__operators__.getitem_24/strided_slice/stack_2?
)tf.__operators__.getitem_24/strided_sliceStridedSlice0graph_conv_38/StatefulPartitionedCall_1:output:08tf.__operators__.getitem_24/strided_slice/stack:output:0:tf.__operators__.getitem_24/strided_slice/stack_1:output:0:tf.__operators__.getitem_24/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2+
)tf.__operators__.getitem_24/strided_slice?
$attention_12/StatefulPartitionedCallStatefulPartitionedCall2tf.__operators__.getitem_24/strided_slice:output:0attention_12_4263640*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_attention_12_layer_call_and_return_conditional_losses_42627292&
$attention_12/StatefulPartitionedCall?
&attention_12/StatefulPartitionedCall_1StatefulPartitionedCall2tf.__operators__.getitem_25/strided_slice:output:0attention_12_4263640*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_attention_12_layer_call_and_return_conditional_losses_42627292(
&attention_12/StatefulPartitionedCall_1?
.neural_tensor_layer_12/StatefulPartitionedCallStatefulPartitionedCall-attention_12/StatefulPartitionedCall:output:0/attention_12/StatefulPartitionedCall_1:output:0neural_tensor_layer_12_4263645neural_tensor_layer_12_4263647neural_tensor_layer_12_4263649*
Tin	
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_neural_tensor_layer_12_layer_call_and_return_conditional_losses_426293520
.neural_tensor_layer_12/StatefulPartitionedCall?
 dense_48/StatefulPartitionedCallStatefulPartitionedCall7neural_tensor_layer_12/StatefulPartitionedCall:output:0dense_48_4263652dense_48_4263654*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_48_layer_call_and_return_conditional_losses_42629542"
 dense_48/StatefulPartitionedCall?
 dense_49/StatefulPartitionedCallStatefulPartitionedCall)dense_48/StatefulPartitionedCall:output:0dense_49_4263657dense_49_4263659*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_49_layer_call_and_return_conditional_losses_42629712"
 dense_49/StatefulPartitionedCall?
 dense_50/StatefulPartitionedCallStatefulPartitionedCall)dense_49/StatefulPartitionedCall:output:0dense_50_4263662dense_50_4263664*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_50_layer_call_and_return_conditional_losses_42629882"
 dense_50/StatefulPartitionedCall?
 dense_51/StatefulPartitionedCallStatefulPartitionedCall)dense_50/StatefulPartitionedCall:output:0dense_51_4263667dense_51_4263669*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_51_layer_call_and_return_conditional_losses_42630042"
 dense_51/StatefulPartitionedCall?
tf.math.sigmoid_12/SigmoidSigmoid)dense_51/StatefulPartitionedCall:output:0*
T0*
_output_shapes

:2
tf.math.sigmoid_12/Sigmoidp
IdentityIdentitytf.math.sigmoid_12/Sigmoid:y:0^NoOp*
T0*
_output_shapes

:2

Identity?
NoOpNoOp%^attention_12/StatefulPartitionedCall'^attention_12/StatefulPartitionedCall_1!^dense_48/StatefulPartitionedCall!^dense_49/StatefulPartitionedCall!^dense_50/StatefulPartitionedCall!^dense_51/StatefulPartitionedCall&^graph_conv_36/StatefulPartitionedCall(^graph_conv_36/StatefulPartitionedCall_1&^graph_conv_37/StatefulPartitionedCall(^graph_conv_37/StatefulPartitionedCall_1&^graph_conv_38/StatefulPartitionedCall(^graph_conv_38/StatefulPartitionedCall_1/^neural_tensor_layer_12/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????	:'???????????????????????????:?????????	:'???????????????????????????: : : : : : : : : : : : : : : : : : 2L
$attention_12/StatefulPartitionedCall$attention_12/StatefulPartitionedCall2P
&attention_12/StatefulPartitionedCall_1&attention_12/StatefulPartitionedCall_12D
 dense_48/StatefulPartitionedCall dense_48/StatefulPartitionedCall2D
 dense_49/StatefulPartitionedCall dense_49/StatefulPartitionedCall2D
 dense_50/StatefulPartitionedCall dense_50/StatefulPartitionedCall2D
 dense_51/StatefulPartitionedCall dense_51/StatefulPartitionedCall2N
%graph_conv_36/StatefulPartitionedCall%graph_conv_36/StatefulPartitionedCall2R
'graph_conv_36/StatefulPartitionedCall_1'graph_conv_36/StatefulPartitionedCall_12N
%graph_conv_37/StatefulPartitionedCall%graph_conv_37/StatefulPartitionedCall2R
'graph_conv_37/StatefulPartitionedCall_1'graph_conv_37/StatefulPartitionedCall_12N
%graph_conv_38/StatefulPartitionedCall%graph_conv_38/StatefulPartitionedCall2R
'graph_conv_38/StatefulPartitionedCall_1'graph_conv_38/StatefulPartitionedCall_12`
.neural_tensor_layer_12/StatefulPartitionedCall.neural_tensor_layer_12/StatefulPartitionedCall:U Q
+
_output_shapes
:?????????	
"
_user_specified_name
input_49:gc
=
_output_shapes+
):'???????????????????????????
"
_user_specified_name
input_50:UQ
+
_output_shapes
:?????????	
"
_user_specified_name
input_51:gc
=
_output_shapes+
):'???????????????????????????
"
_user_specified_name
input_52
?	
?
/__inference_graph_conv_37_layer_call_fn_4265028
inputs_0
inputs_1
unknown:@ 
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :?????????????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_graph_conv_37_layer_call_and_return_conditional_losses_42632472
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :?????????????????? 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*`
_input_shapesO
M:??????????????????@:'???????????????????????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :??????????????????@
"
_user_specified_name
inputs/0:gc
=
_output_shapes+
):'???????????????????????????
"
_user_specified_name
inputs/1
?	
?
E__inference_dense_51_layer_call_and_return_conditional_losses_4263004

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOpj
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpx
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2	
BiasAddb
IdentityIdentityBiasAdd:output:0^NoOp*
T0*
_output_shapes

:2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
:: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:F B

_output_shapes

:
 
_user_specified_nameinputs
??
?
E__inference_model_12_layer_call_and_return_conditional_losses_4264349
inputs_0
inputs_1
inputs_2
inputs_3?
-graph_conv_36_shape_2_readvariableop_resource:@9
+graph_conv_36_add_1_readvariableop_resource:@?
-graph_conv_37_shape_2_readvariableop_resource:@ 9
+graph_conv_37_add_1_readvariableop_resource: ?
-graph_conv_38_shape_2_readvariableop_resource: 9
+graph_conv_38_add_1_readvariableop_resource:=
+attention_12_matmul_readvariableop_resource:G
5neural_tensor_layer_12_matmul_readvariableop_resource: D
.neural_tensor_layer_12_readvariableop_resource:@
2neural_tensor_layer_12_add_readvariableop_resource:9
'dense_48_matmul_readvariableop_resource:6
(dense_48_biasadd_readvariableop_resource:9
'dense_49_matmul_readvariableop_resource:6
(dense_49_biasadd_readvariableop_resource:9
'dense_50_matmul_readvariableop_resource:6
(dense_50_biasadd_readvariableop_resource:9
'dense_51_matmul_readvariableop_resource:6
(dense_51_biasadd_readvariableop_resource:
identity??"attention_12/MatMul/ReadVariableOp?$attention_12/MatMul_3/ReadVariableOp?dense_48/BiasAdd/ReadVariableOp?dense_48/MatMul/ReadVariableOp?dense_49/BiasAdd/ReadVariableOp?dense_49/MatMul/ReadVariableOp?dense_50/BiasAdd/ReadVariableOp?dense_50/MatMul/ReadVariableOp?dense_51/BiasAdd/ReadVariableOp?dense_51/MatMul/ReadVariableOp?"graph_conv_36/add_1/ReadVariableOp?"graph_conv_36/add_5/ReadVariableOp?&graph_conv_36/transpose/ReadVariableOp?(graph_conv_36/transpose_2/ReadVariableOp?"graph_conv_37/add_1/ReadVariableOp?"graph_conv_37/add_5/ReadVariableOp?&graph_conv_37/transpose/ReadVariableOp?(graph_conv_37/transpose_2/ReadVariableOp?"graph_conv_38/add_1/ReadVariableOp?"graph_conv_38/add_5/ReadVariableOp?&graph_conv_38/transpose/ReadVariableOp?(graph_conv_38/transpose_2/ReadVariableOp?,neural_tensor_layer_12/MatMul/ReadVariableOp?%neural_tensor_layer_12/ReadVariableOp?'neural_tensor_layer_12/ReadVariableOp_1?(neural_tensor_layer_12/ReadVariableOp_10?(neural_tensor_layer_12/ReadVariableOp_11?(neural_tensor_layer_12/ReadVariableOp_12?(neural_tensor_layer_12/ReadVariableOp_13?(neural_tensor_layer_12/ReadVariableOp_14?(neural_tensor_layer_12/ReadVariableOp_15?'neural_tensor_layer_12/ReadVariableOp_2?'neural_tensor_layer_12/ReadVariableOp_3?'neural_tensor_layer_12/ReadVariableOp_4?'neural_tensor_layer_12/ReadVariableOp_5?'neural_tensor_layer_12/ReadVariableOp_6?'neural_tensor_layer_12/ReadVariableOp_7?'neural_tensor_layer_12/ReadVariableOp_8?'neural_tensor_layer_12/ReadVariableOp_9?)neural_tensor_layer_12/add/ReadVariableOp?+neural_tensor_layer_12/add_1/ReadVariableOp?,neural_tensor_layer_12/add_10/ReadVariableOp?,neural_tensor_layer_12/add_11/ReadVariableOp?,neural_tensor_layer_12/add_12/ReadVariableOp?,neural_tensor_layer_12/add_13/ReadVariableOp?,neural_tensor_layer_12/add_14/ReadVariableOp?,neural_tensor_layer_12/add_15/ReadVariableOp?+neural_tensor_layer_12/add_2/ReadVariableOp?+neural_tensor_layer_12/add_3/ReadVariableOp?+neural_tensor_layer_12/add_4/ReadVariableOp?+neural_tensor_layer_12/add_5/ReadVariableOp?+neural_tensor_layer_12/add_6/ReadVariableOp?+neural_tensor_layer_12/add_7/ReadVariableOp?+neural_tensor_layer_12/add_8/ReadVariableOp?+neural_tensor_layer_12/add_9/ReadVariableOp?
graph_conv_36/MatMulBatchMatMulV2inputs_3inputs_3*
T0*=
_output_shapes+
):'???????????????????????????2
graph_conv_36/MatMulw
graph_conv_36/ShapeShapegraph_conv_36/MatMul:output:0*
T0*
_output_shapes
:2
graph_conv_36/Shape?
graph_conv_36/addAddV2graph_conv_36/MatMul:output:0inputs_3*
T0*=
_output_shapes+
):'???????????????????????????2
graph_conv_36/addw
graph_conv_36/Greater/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
graph_conv_36/Greater/y?
graph_conv_36/GreaterGreatergraph_conv_36/add:z:0 graph_conv_36/Greater/y:output:0*
T0*=
_output_shapes+
):'???????????????????????????2
graph_conv_36/Greater?
graph_conv_36/CastCastgraph_conv_36/Greater:z:0*

DstT0*

SrcT0
*=
_output_shapes+
):'???????????????????????????2
graph_conv_36/Castf
graph_conv_36/Shape_1Shapeinputs_2*
T0*
_output_shapes
:2
graph_conv_36/Shape_1?
graph_conv_36/unstackUnpackgraph_conv_36/Shape_1:output:0*
T0*
_output_shapes
: : : *	
num2
graph_conv_36/unstack?
$graph_conv_36/Shape_2/ReadVariableOpReadVariableOp-graph_conv_36_shape_2_readvariableop_resource*
_output_shapes

:@*
dtype02&
$graph_conv_36/Shape_2/ReadVariableOp
graph_conv_36/Shape_2Const*
_output_shapes
:*
dtype0*
valueB"   @   2
graph_conv_36/Shape_2?
graph_conv_36/unstack_1Unpackgraph_conv_36/Shape_2:output:0*
T0*
_output_shapes
: : *	
num2
graph_conv_36/unstack_1?
graph_conv_36/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
graph_conv_36/Reshape/shape?
graph_conv_36/ReshapeReshapeinputs_2$graph_conv_36/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????2
graph_conv_36/Reshape?
&graph_conv_36/transpose/ReadVariableOpReadVariableOp-graph_conv_36_shape_2_readvariableop_resource*
_output_shapes

:@*
dtype02(
&graph_conv_36/transpose/ReadVariableOp?
graph_conv_36/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2
graph_conv_36/transpose/perm?
graph_conv_36/transpose	Transpose.graph_conv_36/transpose/ReadVariableOp:value:0%graph_conv_36/transpose/perm:output:0*
T0*
_output_shapes

:@2
graph_conv_36/transpose?
graph_conv_36/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ????2
graph_conv_36/Reshape_1/shape?
graph_conv_36/Reshape_1Reshapegraph_conv_36/transpose:y:0&graph_conv_36/Reshape_1/shape:output:0*
T0*
_output_shapes

:@2
graph_conv_36/Reshape_1?
graph_conv_36/MatMul_1MatMulgraph_conv_36/Reshape:output:0 graph_conv_36/Reshape_1:output:0*
T0*'
_output_shapes
:?????????@2
graph_conv_36/MatMul_1?
graph_conv_36/Reshape_2/shape/1Const*
_output_shapes
: *
dtype0*
value	B :	2!
graph_conv_36/Reshape_2/shape/1?
graph_conv_36/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :@2!
graph_conv_36/Reshape_2/shape/2?
graph_conv_36/Reshape_2/shapePackgraph_conv_36/unstack:output:0(graph_conv_36/Reshape_2/shape/1:output:0(graph_conv_36/Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:2
graph_conv_36/Reshape_2/shape?
graph_conv_36/Reshape_2Reshape graph_conv_36/MatMul_1:product:0&graph_conv_36/Reshape_2/shape:output:0*
T0*+
_output_shapes
:?????????	@2
graph_conv_36/Reshape_2?
"graph_conv_36/add_1/ReadVariableOpReadVariableOp+graph_conv_36_add_1_readvariableop_resource*
_output_shapes
:@*
dtype02$
"graph_conv_36/add_1/ReadVariableOp?
graph_conv_36/add_1AddV2 graph_conv_36/Reshape_2:output:0*graph_conv_36/add_1/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????	@2
graph_conv_36/add_1?
graph_conv_36/MatMul_2BatchMatMulV2graph_conv_36/Cast:y:0graph_conv_36/Cast:y:0*
T0*=
_output_shapes+
):'???????????????????????????2
graph_conv_36/MatMul_2}
graph_conv_36/Shape_3Shapegraph_conv_36/MatMul_2:output:0*
T0*
_output_shapes
:2
graph_conv_36/Shape_3?
graph_conv_36/add_2AddV2graph_conv_36/MatMul_2:output:0graph_conv_36/Cast:y:0*
T0*=
_output_shapes+
):'???????????????????????????2
graph_conv_36/add_2{
graph_conv_36/Greater_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
graph_conv_36/Greater_1/y?
graph_conv_36/Greater_1Greatergraph_conv_36/add_2:z:0"graph_conv_36/Greater_1/y:output:0*
T0*=
_output_shapes+
):'???????????????????????????2
graph_conv_36/Greater_1?
graph_conv_36/Cast_1Castgraph_conv_36/Greater_1:z:0*

DstT0*

SrcT0
*=
_output_shapes+
):'???????????????????????????2
graph_conv_36/Cast_1?
graph_conv_36/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2 
graph_conv_36/transpose_1/perm?
graph_conv_36/transpose_1	Transposegraph_conv_36/Cast_1:y:0'graph_conv_36/transpose_1/perm:output:0*
T0*=
_output_shapes+
):'???????????????????????????2
graph_conv_36/transpose_1?
graph_conv_36/MatMul_3BatchMatMulV2graph_conv_36/transpose_1:y:0graph_conv_36/add_1:z:0*
T0*4
_output_shapes"
 :??????????????????@2
graph_conv_36/MatMul_3}
graph_conv_36/Shape_4Shapegraph_conv_36/MatMul_3:output:0*
T0*
_output_shapes
:2
graph_conv_36/Shape_4?
#graph_conv_36/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2%
#graph_conv_36/Sum/reduction_indices?
graph_conv_36/SumSumgraph_conv_36/Cast_1:y:0,graph_conv_36/Sum/reduction_indices:output:0*
T0*4
_output_shapes"
 :??????????????????*
	keep_dims(2
graph_conv_36/Sums
graph_conv_36/add_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
graph_conv_36/add_3/y?
graph_conv_36/add_3AddV2graph_conv_36/Sum:output:0graph_conv_36/add_3/y:output:0*
T0*4
_output_shapes"
 :??????????????????2
graph_conv_36/add_3?
graph_conv_36/truedivRealDivgraph_conv_36/MatMul_3:output:0graph_conv_36/add_3:z:0*
T0*4
_output_shapes"
 :??????????????????@2
graph_conv_36/truediv?
graph_conv_36/ReluRelugraph_conv_36/truediv:z:0*
T0*4
_output_shapes"
 :??????????????????@2
graph_conv_36/Relu?
graph_conv_36/MatMul_4BatchMatMulV2inputs_1inputs_1*
T0*=
_output_shapes+
):'???????????????????????????2
graph_conv_36/MatMul_4}
graph_conv_36/Shape_5Shapegraph_conv_36/MatMul_4:output:0*
T0*
_output_shapes
:2
graph_conv_36/Shape_5?
graph_conv_36/add_4AddV2graph_conv_36/MatMul_4:output:0inputs_1*
T0*=
_output_shapes+
):'???????????????????????????2
graph_conv_36/add_4{
graph_conv_36/Greater_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
graph_conv_36/Greater_2/y?
graph_conv_36/Greater_2Greatergraph_conv_36/add_4:z:0"graph_conv_36/Greater_2/y:output:0*
T0*=
_output_shapes+
):'???????????????????????????2
graph_conv_36/Greater_2?
graph_conv_36/Cast_2Castgraph_conv_36/Greater_2:z:0*

DstT0*

SrcT0
*=
_output_shapes+
):'???????????????????????????2
graph_conv_36/Cast_2f
graph_conv_36/Shape_6Shapeinputs_0*
T0*
_output_shapes
:2
graph_conv_36/Shape_6?
graph_conv_36/unstack_2Unpackgraph_conv_36/Shape_6:output:0*
T0*
_output_shapes
: : : *	
num2
graph_conv_36/unstack_2?
$graph_conv_36/Shape_7/ReadVariableOpReadVariableOp-graph_conv_36_shape_2_readvariableop_resource*
_output_shapes

:@*
dtype02&
$graph_conv_36/Shape_7/ReadVariableOp
graph_conv_36/Shape_7Const*
_output_shapes
:*
dtype0*
valueB"   @   2
graph_conv_36/Shape_7?
graph_conv_36/unstack_3Unpackgraph_conv_36/Shape_7:output:0*
T0*
_output_shapes
: : *	
num2
graph_conv_36/unstack_3?
graph_conv_36/Reshape_3/shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2
graph_conv_36/Reshape_3/shape?
graph_conv_36/Reshape_3Reshapeinputs_0&graph_conv_36/Reshape_3/shape:output:0*
T0*'
_output_shapes
:?????????2
graph_conv_36/Reshape_3?
(graph_conv_36/transpose_2/ReadVariableOpReadVariableOp-graph_conv_36_shape_2_readvariableop_resource*
_output_shapes

:@*
dtype02*
(graph_conv_36/transpose_2/ReadVariableOp?
graph_conv_36/transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2 
graph_conv_36/transpose_2/perm?
graph_conv_36/transpose_2	Transpose0graph_conv_36/transpose_2/ReadVariableOp:value:0'graph_conv_36/transpose_2/perm:output:0*
T0*
_output_shapes

:@2
graph_conv_36/transpose_2?
graph_conv_36/Reshape_4/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ????2
graph_conv_36/Reshape_4/shape?
graph_conv_36/Reshape_4Reshapegraph_conv_36/transpose_2:y:0&graph_conv_36/Reshape_4/shape:output:0*
T0*
_output_shapes

:@2
graph_conv_36/Reshape_4?
graph_conv_36/MatMul_5MatMul graph_conv_36/Reshape_3:output:0 graph_conv_36/Reshape_4:output:0*
T0*'
_output_shapes
:?????????@2
graph_conv_36/MatMul_5?
graph_conv_36/Reshape_5/shape/1Const*
_output_shapes
: *
dtype0*
value	B :	2!
graph_conv_36/Reshape_5/shape/1?
graph_conv_36/Reshape_5/shape/2Const*
_output_shapes
: *
dtype0*
value	B :@2!
graph_conv_36/Reshape_5/shape/2?
graph_conv_36/Reshape_5/shapePack graph_conv_36/unstack_2:output:0(graph_conv_36/Reshape_5/shape/1:output:0(graph_conv_36/Reshape_5/shape/2:output:0*
N*
T0*
_output_shapes
:2
graph_conv_36/Reshape_5/shape?
graph_conv_36/Reshape_5Reshape graph_conv_36/MatMul_5:product:0&graph_conv_36/Reshape_5/shape:output:0*
T0*+
_output_shapes
:?????????	@2
graph_conv_36/Reshape_5?
"graph_conv_36/add_5/ReadVariableOpReadVariableOp+graph_conv_36_add_1_readvariableop_resource*
_output_shapes
:@*
dtype02$
"graph_conv_36/add_5/ReadVariableOp?
graph_conv_36/add_5AddV2 graph_conv_36/Reshape_5:output:0*graph_conv_36/add_5/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????	@2
graph_conv_36/add_5?
graph_conv_36/MatMul_6BatchMatMulV2graph_conv_36/Cast_2:y:0graph_conv_36/Cast_2:y:0*
T0*=
_output_shapes+
):'???????????????????????????2
graph_conv_36/MatMul_6}
graph_conv_36/Shape_8Shapegraph_conv_36/MatMul_6:output:0*
T0*
_output_shapes
:2
graph_conv_36/Shape_8?
graph_conv_36/add_6AddV2graph_conv_36/MatMul_6:output:0graph_conv_36/Cast_2:y:0*
T0*=
_output_shapes+
):'???????????????????????????2
graph_conv_36/add_6{
graph_conv_36/Greater_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
graph_conv_36/Greater_3/y?
graph_conv_36/Greater_3Greatergraph_conv_36/add_6:z:0"graph_conv_36/Greater_3/y:output:0*
T0*=
_output_shapes+
):'???????????????????????????2
graph_conv_36/Greater_3?
graph_conv_36/Cast_3Castgraph_conv_36/Greater_3:z:0*

DstT0*

SrcT0
*=
_output_shapes+
):'???????????????????????????2
graph_conv_36/Cast_3?
graph_conv_36/transpose_3/permConst*
_output_shapes
:*
dtype0*!
valueB"          2 
graph_conv_36/transpose_3/perm?
graph_conv_36/transpose_3	Transposegraph_conv_36/Cast_3:y:0'graph_conv_36/transpose_3/perm:output:0*
T0*=
_output_shapes+
):'???????????????????????????2
graph_conv_36/transpose_3?
graph_conv_36/MatMul_7BatchMatMulV2graph_conv_36/transpose_3:y:0graph_conv_36/add_5:z:0*
T0*4
_output_shapes"
 :??????????????????@2
graph_conv_36/MatMul_7}
graph_conv_36/Shape_9Shapegraph_conv_36/MatMul_7:output:0*
T0*
_output_shapes
:2
graph_conv_36/Shape_9?
%graph_conv_36/Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2'
%graph_conv_36/Sum_1/reduction_indices?
graph_conv_36/Sum_1Sumgraph_conv_36/Cast_3:y:0.graph_conv_36/Sum_1/reduction_indices:output:0*
T0*4
_output_shapes"
 :??????????????????*
	keep_dims(2
graph_conv_36/Sum_1s
graph_conv_36/add_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
graph_conv_36/add_7/y?
graph_conv_36/add_7AddV2graph_conv_36/Sum_1:output:0graph_conv_36/add_7/y:output:0*
T0*4
_output_shapes"
 :??????????????????2
graph_conv_36/add_7?
graph_conv_36/truediv_1RealDivgraph_conv_36/MatMul_7:output:0graph_conv_36/add_7:z:0*
T0*4
_output_shapes"
 :??????????????????@2
graph_conv_36/truediv_1?
graph_conv_36/Relu_1Relugraph_conv_36/truediv_1:z:0*
T0*4
_output_shapes"
 :??????????????????@2
graph_conv_36/Relu_1?
graph_conv_37/MatMulBatchMatMulV2inputs_3inputs_3*
T0*=
_output_shapes+
):'???????????????????????????2
graph_conv_37/MatMulw
graph_conv_37/ShapeShapegraph_conv_37/MatMul:output:0*
T0*
_output_shapes
:2
graph_conv_37/Shape?
graph_conv_37/addAddV2graph_conv_37/MatMul:output:0inputs_3*
T0*=
_output_shapes+
):'???????????????????????????2
graph_conv_37/addw
graph_conv_37/Greater/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
graph_conv_37/Greater/y?
graph_conv_37/GreaterGreatergraph_conv_37/add:z:0 graph_conv_37/Greater/y:output:0*
T0*=
_output_shapes+
):'???????????????????????????2
graph_conv_37/Greater?
graph_conv_37/CastCastgraph_conv_37/Greater:z:0*

DstT0*

SrcT0
*=
_output_shapes+
):'???????????????????????????2
graph_conv_37/Cast~
graph_conv_37/Shape_1Shape graph_conv_36/Relu:activations:0*
T0*
_output_shapes
:2
graph_conv_37/Shape_1?
graph_conv_37/unstackUnpackgraph_conv_37/Shape_1:output:0*
T0*
_output_shapes
: : : *	
num2
graph_conv_37/unstack?
$graph_conv_37/Shape_2/ReadVariableOpReadVariableOp-graph_conv_37_shape_2_readvariableop_resource*
_output_shapes

:@ *
dtype02&
$graph_conv_37/Shape_2/ReadVariableOp
graph_conv_37/Shape_2Const*
_output_shapes
:*
dtype0*
valueB"@       2
graph_conv_37/Shape_2?
graph_conv_37/unstack_1Unpackgraph_conv_37/Shape_2:output:0*
T0*
_output_shapes
: : *	
num2
graph_conv_37/unstack_1?
graph_conv_37/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   2
graph_conv_37/Reshape/shape?
graph_conv_37/ReshapeReshape graph_conv_36/Relu:activations:0$graph_conv_37/Reshape/shape:output:0*
T0*'
_output_shapes
:?????????@2
graph_conv_37/Reshape?
&graph_conv_37/transpose/ReadVariableOpReadVariableOp-graph_conv_37_shape_2_readvariableop_resource*
_output_shapes

:@ *
dtype02(
&graph_conv_37/transpose/ReadVariableOp?
graph_conv_37/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2
graph_conv_37/transpose/perm?
graph_conv_37/transpose	Transpose.graph_conv_37/transpose/ReadVariableOp:value:0%graph_conv_37/transpose/perm:output:0*
T0*
_output_shapes

:@ 2
graph_conv_37/transpose?
graph_conv_37/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"@   ????2
graph_conv_37/Reshape_1/shape?
graph_conv_37/Reshape_1Reshapegraph_conv_37/transpose:y:0&graph_conv_37/Reshape_1/shape:output:0*
T0*
_output_shapes

:@ 2
graph_conv_37/Reshape_1?
graph_conv_37/MatMul_1MatMulgraph_conv_37/Reshape:output:0 graph_conv_37/Reshape_1:output:0*
T0*'
_output_shapes
:????????? 2
graph_conv_37/MatMul_1?
graph_conv_37/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B : 2!
graph_conv_37/Reshape_2/shape/2?
graph_conv_37/Reshape_2/shapePackgraph_conv_37/unstack:output:0graph_conv_37/unstack:output:1(graph_conv_37/Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:2
graph_conv_37/Reshape_2/shape?
graph_conv_37/Reshape_2Reshape graph_conv_37/MatMul_1:product:0&graph_conv_37/Reshape_2/shape:output:0*
T0*4
_output_shapes"
 :?????????????????? 2
graph_conv_37/Reshape_2?
"graph_conv_37/add_1/ReadVariableOpReadVariableOp+graph_conv_37_add_1_readvariableop_resource*
_output_shapes
: *
dtype02$
"graph_conv_37/add_1/ReadVariableOp?
graph_conv_37/add_1AddV2 graph_conv_37/Reshape_2:output:0*graph_conv_37/add_1/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :?????????????????? 2
graph_conv_37/add_1?
graph_conv_37/MatMul_2BatchMatMulV2graph_conv_37/Cast:y:0graph_conv_37/Cast:y:0*
T0*=
_output_shapes+
):'???????????????????????????2
graph_conv_37/MatMul_2}
graph_conv_37/Shape_3Shapegraph_conv_37/MatMul_2:output:0*
T0*
_output_shapes
:2
graph_conv_37/Shape_3?
graph_conv_37/add_2AddV2graph_conv_37/MatMul_2:output:0graph_conv_37/Cast:y:0*
T0*=
_output_shapes+
):'???????????????????????????2
graph_conv_37/add_2{
graph_conv_37/Greater_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
graph_conv_37/Greater_1/y?
graph_conv_37/Greater_1Greatergraph_conv_37/add_2:z:0"graph_conv_37/Greater_1/y:output:0*
T0*=
_output_shapes+
):'???????????????????????????2
graph_conv_37/Greater_1?
graph_conv_37/Cast_1Castgraph_conv_37/Greater_1:z:0*

DstT0*

SrcT0
*=
_output_shapes+
):'???????????????????????????2
graph_conv_37/Cast_1?
graph_conv_37/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2 
graph_conv_37/transpose_1/perm?
graph_conv_37/transpose_1	Transposegraph_conv_37/Cast_1:y:0'graph_conv_37/transpose_1/perm:output:0*
T0*=
_output_shapes+
):'???????????????????????????2
graph_conv_37/transpose_1?
graph_conv_37/MatMul_3BatchMatMulV2graph_conv_37/transpose_1:y:0graph_conv_37/add_1:z:0*
T0*4
_output_shapes"
 :?????????????????? 2
graph_conv_37/MatMul_3}
graph_conv_37/Shape_4Shapegraph_conv_37/MatMul_3:output:0*
T0*
_output_shapes
:2
graph_conv_37/Shape_4?
#graph_conv_37/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2%
#graph_conv_37/Sum/reduction_indices?
graph_conv_37/SumSumgraph_conv_37/Cast_1:y:0,graph_conv_37/Sum/reduction_indices:output:0*
T0*4
_output_shapes"
 :??????????????????*
	keep_dims(2
graph_conv_37/Sums
graph_conv_37/add_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
graph_conv_37/add_3/y?
graph_conv_37/add_3AddV2graph_conv_37/Sum:output:0graph_conv_37/add_3/y:output:0*
T0*4
_output_shapes"
 :??????????????????2
graph_conv_37/add_3?
graph_conv_37/truedivRealDivgraph_conv_37/MatMul_3:output:0graph_conv_37/add_3:z:0*
T0*4
_output_shapes"
 :?????????????????? 2
graph_conv_37/truediv?
graph_conv_37/ReluRelugraph_conv_37/truediv:z:0*
T0*4
_output_shapes"
 :?????????????????? 2
graph_conv_37/Relu?
graph_conv_37/MatMul_4BatchMatMulV2inputs_1inputs_1*
T0*=
_output_shapes+
):'???????????????????????????2
graph_conv_37/MatMul_4}
graph_conv_37/Shape_5Shapegraph_conv_37/MatMul_4:output:0*
T0*
_output_shapes
:2
graph_conv_37/Shape_5?
graph_conv_37/add_4AddV2graph_conv_37/MatMul_4:output:0inputs_1*
T0*=
_output_shapes+
):'???????????????????????????2
graph_conv_37/add_4{
graph_conv_37/Greater_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
graph_conv_37/Greater_2/y?
graph_conv_37/Greater_2Greatergraph_conv_37/add_4:z:0"graph_conv_37/Greater_2/y:output:0*
T0*=
_output_shapes+
):'???????????????????????????2
graph_conv_37/Greater_2?
graph_conv_37/Cast_2Castgraph_conv_37/Greater_2:z:0*

DstT0*

SrcT0
*=
_output_shapes+
):'???????????????????????????2
graph_conv_37/Cast_2?
graph_conv_37/Shape_6Shape"graph_conv_36/Relu_1:activations:0*
T0*
_output_shapes
:2
graph_conv_37/Shape_6?
graph_conv_37/unstack_2Unpackgraph_conv_37/Shape_6:output:0*
T0*
_output_shapes
: : : *	
num2
graph_conv_37/unstack_2?
$graph_conv_37/Shape_7/ReadVariableOpReadVariableOp-graph_conv_37_shape_2_readvariableop_resource*
_output_shapes

:@ *
dtype02&
$graph_conv_37/Shape_7/ReadVariableOp
graph_conv_37/Shape_7Const*
_output_shapes
:*
dtype0*
valueB"@       2
graph_conv_37/Shape_7?
graph_conv_37/unstack_3Unpackgraph_conv_37/Shape_7:output:0*
T0*
_output_shapes
: : *	
num2
graph_conv_37/unstack_3?
graph_conv_37/Reshape_3/shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   2
graph_conv_37/Reshape_3/shape?
graph_conv_37/Reshape_3Reshape"graph_conv_36/Relu_1:activations:0&graph_conv_37/Reshape_3/shape:output:0*
T0*'
_output_shapes
:?????????@2
graph_conv_37/Reshape_3?
(graph_conv_37/transpose_2/ReadVariableOpReadVariableOp-graph_conv_37_shape_2_readvariableop_resource*
_output_shapes

:@ *
dtype02*
(graph_conv_37/transpose_2/ReadVariableOp?
graph_conv_37/transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2 
graph_conv_37/transpose_2/perm?
graph_conv_37/transpose_2	Transpose0graph_conv_37/transpose_2/ReadVariableOp:value:0'graph_conv_37/transpose_2/perm:output:0*
T0*
_output_shapes

:@ 2
graph_conv_37/transpose_2?
graph_conv_37/Reshape_4/shapeConst*
_output_shapes
:*
dtype0*
valueB"@   ????2
graph_conv_37/Reshape_4/shape?
graph_conv_37/Reshape_4Reshapegraph_conv_37/transpose_2:y:0&graph_conv_37/Reshape_4/shape:output:0*
T0*
_output_shapes

:@ 2
graph_conv_37/Reshape_4?
graph_conv_37/MatMul_5MatMul graph_conv_37/Reshape_3:output:0 graph_conv_37/Reshape_4:output:0*
T0*'
_output_shapes
:????????? 2
graph_conv_37/MatMul_5?
graph_conv_37/Reshape_5/shape/2Const*
_output_shapes
: *
dtype0*
value	B : 2!
graph_conv_37/Reshape_5/shape/2?
graph_conv_37/Reshape_5/shapePack graph_conv_37/unstack_2:output:0 graph_conv_37/unstack_2:output:1(graph_conv_37/Reshape_5/shape/2:output:0*
N*
T0*
_output_shapes
:2
graph_conv_37/Reshape_5/shape?
graph_conv_37/Reshape_5Reshape graph_conv_37/MatMul_5:product:0&graph_conv_37/Reshape_5/shape:output:0*
T0*4
_output_shapes"
 :?????????????????? 2
graph_conv_37/Reshape_5?
"graph_conv_37/add_5/ReadVariableOpReadVariableOp+graph_conv_37_add_1_readvariableop_resource*
_output_shapes
: *
dtype02$
"graph_conv_37/add_5/ReadVariableOp?
graph_conv_37/add_5AddV2 graph_conv_37/Reshape_5:output:0*graph_conv_37/add_5/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :?????????????????? 2
graph_conv_37/add_5?
graph_conv_37/MatMul_6BatchMatMulV2graph_conv_37/Cast_2:y:0graph_conv_37/Cast_2:y:0*
T0*=
_output_shapes+
):'???????????????????????????2
graph_conv_37/MatMul_6}
graph_conv_37/Shape_8Shapegraph_conv_37/MatMul_6:output:0*
T0*
_output_shapes
:2
graph_conv_37/Shape_8?
graph_conv_37/add_6AddV2graph_conv_37/MatMul_6:output:0graph_conv_37/Cast_2:y:0*
T0*=
_output_shapes+
):'???????????????????????????2
graph_conv_37/add_6{
graph_conv_37/Greater_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
graph_conv_37/Greater_3/y?
graph_conv_37/Greater_3Greatergraph_conv_37/add_6:z:0"graph_conv_37/Greater_3/y:output:0*
T0*=
_output_shapes+
):'???????????????????????????2
graph_conv_37/Greater_3?
graph_conv_37/Cast_3Castgraph_conv_37/Greater_3:z:0*

DstT0*

SrcT0
*=
_output_shapes+
):'???????????????????????????2
graph_conv_37/Cast_3?
graph_conv_37/transpose_3/permConst*
_output_shapes
:*
dtype0*!
valueB"          2 
graph_conv_37/transpose_3/perm?
graph_conv_37/transpose_3	Transposegraph_conv_37/Cast_3:y:0'graph_conv_37/transpose_3/perm:output:0*
T0*=
_output_shapes+
):'???????????????????????????2
graph_conv_37/transpose_3?
graph_conv_37/MatMul_7BatchMatMulV2graph_conv_37/transpose_3:y:0graph_conv_37/add_5:z:0*
T0*4
_output_shapes"
 :?????????????????? 2
graph_conv_37/MatMul_7}
graph_conv_37/Shape_9Shapegraph_conv_37/MatMul_7:output:0*
T0*
_output_shapes
:2
graph_conv_37/Shape_9?
%graph_conv_37/Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2'
%graph_conv_37/Sum_1/reduction_indices?
graph_conv_37/Sum_1Sumgraph_conv_37/Cast_3:y:0.graph_conv_37/Sum_1/reduction_indices:output:0*
T0*4
_output_shapes"
 :??????????????????*
	keep_dims(2
graph_conv_37/Sum_1s
graph_conv_37/add_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
graph_conv_37/add_7/y?
graph_conv_37/add_7AddV2graph_conv_37/Sum_1:output:0graph_conv_37/add_7/y:output:0*
T0*4
_output_shapes"
 :??????????????????2
graph_conv_37/add_7?
graph_conv_37/truediv_1RealDivgraph_conv_37/MatMul_7:output:0graph_conv_37/add_7:z:0*
T0*4
_output_shapes"
 :?????????????????? 2
graph_conv_37/truediv_1?
graph_conv_37/Relu_1Relugraph_conv_37/truediv_1:z:0*
T0*4
_output_shapes"
 :?????????????????? 2
graph_conv_37/Relu_1?
graph_conv_38/MatMulBatchMatMulV2inputs_3inputs_3*
T0*=
_output_shapes+
):'???????????????????????????2
graph_conv_38/MatMulw
graph_conv_38/ShapeShapegraph_conv_38/MatMul:output:0*
T0*
_output_shapes
:2
graph_conv_38/Shape?
graph_conv_38/addAddV2graph_conv_38/MatMul:output:0inputs_3*
T0*=
_output_shapes+
):'???????????????????????????2
graph_conv_38/addw
graph_conv_38/Greater/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
graph_conv_38/Greater/y?
graph_conv_38/GreaterGreatergraph_conv_38/add:z:0 graph_conv_38/Greater/y:output:0*
T0*=
_output_shapes+
):'???????????????????????????2
graph_conv_38/Greater?
graph_conv_38/CastCastgraph_conv_38/Greater:z:0*

DstT0*

SrcT0
*=
_output_shapes+
):'???????????????????????????2
graph_conv_38/Cast~
graph_conv_38/Shape_1Shape graph_conv_37/Relu:activations:0*
T0*
_output_shapes
:2
graph_conv_38/Shape_1?
graph_conv_38/unstackUnpackgraph_conv_38/Shape_1:output:0*
T0*
_output_shapes
: : : *	
num2
graph_conv_38/unstack?
$graph_conv_38/Shape_2/ReadVariableOpReadVariableOp-graph_conv_38_shape_2_readvariableop_resource*
_output_shapes

: *
dtype02&
$graph_conv_38/Shape_2/ReadVariableOp
graph_conv_38/Shape_2Const*
_output_shapes
:*
dtype0*
valueB"       2
graph_conv_38/Shape_2?
graph_conv_38/unstack_1Unpackgraph_conv_38/Shape_2:output:0*
T0*
_output_shapes
: : *	
num2
graph_conv_38/unstack_1?
graph_conv_38/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????    2
graph_conv_38/Reshape/shape?
graph_conv_38/ReshapeReshape graph_conv_37/Relu:activations:0$graph_conv_38/Reshape/shape:output:0*
T0*'
_output_shapes
:????????? 2
graph_conv_38/Reshape?
&graph_conv_38/transpose/ReadVariableOpReadVariableOp-graph_conv_38_shape_2_readvariableop_resource*
_output_shapes

: *
dtype02(
&graph_conv_38/transpose/ReadVariableOp?
graph_conv_38/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2
graph_conv_38/transpose/perm?
graph_conv_38/transpose	Transpose.graph_conv_38/transpose/ReadVariableOp:value:0%graph_conv_38/transpose/perm:output:0*
T0*
_output_shapes

: 2
graph_conv_38/transpose?
graph_conv_38/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"    ????2
graph_conv_38/Reshape_1/shape?
graph_conv_38/Reshape_1Reshapegraph_conv_38/transpose:y:0&graph_conv_38/Reshape_1/shape:output:0*
T0*
_output_shapes

: 2
graph_conv_38/Reshape_1?
graph_conv_38/MatMul_1MatMulgraph_conv_38/Reshape:output:0 graph_conv_38/Reshape_1:output:0*
T0*'
_output_shapes
:?????????2
graph_conv_38/MatMul_1?
graph_conv_38/Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2!
graph_conv_38/Reshape_2/shape/2?
graph_conv_38/Reshape_2/shapePackgraph_conv_38/unstack:output:0graph_conv_38/unstack:output:1(graph_conv_38/Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:2
graph_conv_38/Reshape_2/shape?
graph_conv_38/Reshape_2Reshape graph_conv_38/MatMul_1:product:0&graph_conv_38/Reshape_2/shape:output:0*
T0*4
_output_shapes"
 :??????????????????2
graph_conv_38/Reshape_2?
"graph_conv_38/add_1/ReadVariableOpReadVariableOp+graph_conv_38_add_1_readvariableop_resource*
_output_shapes
:*
dtype02$
"graph_conv_38/add_1/ReadVariableOp?
graph_conv_38/add_1AddV2 graph_conv_38/Reshape_2:output:0*graph_conv_38/add_1/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :??????????????????2
graph_conv_38/add_1?
graph_conv_38/MatMul_2BatchMatMulV2graph_conv_38/Cast:y:0graph_conv_38/Cast:y:0*
T0*=
_output_shapes+
):'???????????????????????????2
graph_conv_38/MatMul_2}
graph_conv_38/Shape_3Shapegraph_conv_38/MatMul_2:output:0*
T0*
_output_shapes
:2
graph_conv_38/Shape_3?
graph_conv_38/add_2AddV2graph_conv_38/MatMul_2:output:0graph_conv_38/Cast:y:0*
T0*=
_output_shapes+
):'???????????????????????????2
graph_conv_38/add_2{
graph_conv_38/Greater_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
graph_conv_38/Greater_1/y?
graph_conv_38/Greater_1Greatergraph_conv_38/add_2:z:0"graph_conv_38/Greater_1/y:output:0*
T0*=
_output_shapes+
):'???????????????????????????2
graph_conv_38/Greater_1?
graph_conv_38/Cast_1Castgraph_conv_38/Greater_1:z:0*

DstT0*

SrcT0
*=
_output_shapes+
):'???????????????????????????2
graph_conv_38/Cast_1?
graph_conv_38/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2 
graph_conv_38/transpose_1/perm?
graph_conv_38/transpose_1	Transposegraph_conv_38/Cast_1:y:0'graph_conv_38/transpose_1/perm:output:0*
T0*=
_output_shapes+
):'???????????????????????????2
graph_conv_38/transpose_1?
graph_conv_38/MatMul_3BatchMatMulV2graph_conv_38/transpose_1:y:0graph_conv_38/add_1:z:0*
T0*4
_output_shapes"
 :??????????????????2
graph_conv_38/MatMul_3}
graph_conv_38/Shape_4Shapegraph_conv_38/MatMul_3:output:0*
T0*
_output_shapes
:2
graph_conv_38/Shape_4?
#graph_conv_38/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2%
#graph_conv_38/Sum/reduction_indices?
graph_conv_38/SumSumgraph_conv_38/Cast_1:y:0,graph_conv_38/Sum/reduction_indices:output:0*
T0*4
_output_shapes"
 :??????????????????*
	keep_dims(2
graph_conv_38/Sums
graph_conv_38/add_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
graph_conv_38/add_3/y?
graph_conv_38/add_3AddV2graph_conv_38/Sum:output:0graph_conv_38/add_3/y:output:0*
T0*4
_output_shapes"
 :??????????????????2
graph_conv_38/add_3?
graph_conv_38/truedivRealDivgraph_conv_38/MatMul_3:output:0graph_conv_38/add_3:z:0*
T0*4
_output_shapes"
 :??????????????????2
graph_conv_38/truediv?
graph_conv_38/ReluRelugraph_conv_38/truediv:z:0*
T0*4
_output_shapes"
 :??????????????????2
graph_conv_38/Relu?
graph_conv_38/MatMul_4BatchMatMulV2inputs_1inputs_1*
T0*=
_output_shapes+
):'???????????????????????????2
graph_conv_38/MatMul_4}
graph_conv_38/Shape_5Shapegraph_conv_38/MatMul_4:output:0*
T0*
_output_shapes
:2
graph_conv_38/Shape_5?
graph_conv_38/add_4AddV2graph_conv_38/MatMul_4:output:0inputs_1*
T0*=
_output_shapes+
):'???????????????????????????2
graph_conv_38/add_4{
graph_conv_38/Greater_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
graph_conv_38/Greater_2/y?
graph_conv_38/Greater_2Greatergraph_conv_38/add_4:z:0"graph_conv_38/Greater_2/y:output:0*
T0*=
_output_shapes+
):'???????????????????????????2
graph_conv_38/Greater_2?
graph_conv_38/Cast_2Castgraph_conv_38/Greater_2:z:0*

DstT0*

SrcT0
*=
_output_shapes+
):'???????????????????????????2
graph_conv_38/Cast_2?
graph_conv_38/Shape_6Shape"graph_conv_37/Relu_1:activations:0*
T0*
_output_shapes
:2
graph_conv_38/Shape_6?
graph_conv_38/unstack_2Unpackgraph_conv_38/Shape_6:output:0*
T0*
_output_shapes
: : : *	
num2
graph_conv_38/unstack_2?
$graph_conv_38/Shape_7/ReadVariableOpReadVariableOp-graph_conv_38_shape_2_readvariableop_resource*
_output_shapes

: *
dtype02&
$graph_conv_38/Shape_7/ReadVariableOp
graph_conv_38/Shape_7Const*
_output_shapes
:*
dtype0*
valueB"       2
graph_conv_38/Shape_7?
graph_conv_38/unstack_3Unpackgraph_conv_38/Shape_7:output:0*
T0*
_output_shapes
: : *	
num2
graph_conv_38/unstack_3?
graph_conv_38/Reshape_3/shapeConst*
_output_shapes
:*
dtype0*
valueB"????    2
graph_conv_38/Reshape_3/shape?
graph_conv_38/Reshape_3Reshape"graph_conv_37/Relu_1:activations:0&graph_conv_38/Reshape_3/shape:output:0*
T0*'
_output_shapes
:????????? 2
graph_conv_38/Reshape_3?
(graph_conv_38/transpose_2/ReadVariableOpReadVariableOp-graph_conv_38_shape_2_readvariableop_resource*
_output_shapes

: *
dtype02*
(graph_conv_38/transpose_2/ReadVariableOp?
graph_conv_38/transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2 
graph_conv_38/transpose_2/perm?
graph_conv_38/transpose_2	Transpose0graph_conv_38/transpose_2/ReadVariableOp:value:0'graph_conv_38/transpose_2/perm:output:0*
T0*
_output_shapes

: 2
graph_conv_38/transpose_2?
graph_conv_38/Reshape_4/shapeConst*
_output_shapes
:*
dtype0*
valueB"    ????2
graph_conv_38/Reshape_4/shape?
graph_conv_38/Reshape_4Reshapegraph_conv_38/transpose_2:y:0&graph_conv_38/Reshape_4/shape:output:0*
T0*
_output_shapes

: 2
graph_conv_38/Reshape_4?
graph_conv_38/MatMul_5MatMul graph_conv_38/Reshape_3:output:0 graph_conv_38/Reshape_4:output:0*
T0*'
_output_shapes
:?????????2
graph_conv_38/MatMul_5?
graph_conv_38/Reshape_5/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2!
graph_conv_38/Reshape_5/shape/2?
graph_conv_38/Reshape_5/shapePack graph_conv_38/unstack_2:output:0 graph_conv_38/unstack_2:output:1(graph_conv_38/Reshape_5/shape/2:output:0*
N*
T0*
_output_shapes
:2
graph_conv_38/Reshape_5/shape?
graph_conv_38/Reshape_5Reshape graph_conv_38/MatMul_5:product:0&graph_conv_38/Reshape_5/shape:output:0*
T0*4
_output_shapes"
 :??????????????????2
graph_conv_38/Reshape_5?
"graph_conv_38/add_5/ReadVariableOpReadVariableOp+graph_conv_38_add_1_readvariableop_resource*
_output_shapes
:*
dtype02$
"graph_conv_38/add_5/ReadVariableOp?
graph_conv_38/add_5AddV2 graph_conv_38/Reshape_5:output:0*graph_conv_38/add_5/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :??????????????????2
graph_conv_38/add_5?
graph_conv_38/MatMul_6BatchMatMulV2graph_conv_38/Cast_2:y:0graph_conv_38/Cast_2:y:0*
T0*=
_output_shapes+
):'???????????????????????????2
graph_conv_38/MatMul_6}
graph_conv_38/Shape_8Shapegraph_conv_38/MatMul_6:output:0*
T0*
_output_shapes
:2
graph_conv_38/Shape_8?
graph_conv_38/add_6AddV2graph_conv_38/MatMul_6:output:0graph_conv_38/Cast_2:y:0*
T0*=
_output_shapes+
):'???????????????????????????2
graph_conv_38/add_6{
graph_conv_38/Greater_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
graph_conv_38/Greater_3/y?
graph_conv_38/Greater_3Greatergraph_conv_38/add_6:z:0"graph_conv_38/Greater_3/y:output:0*
T0*=
_output_shapes+
):'???????????????????????????2
graph_conv_38/Greater_3?
graph_conv_38/Cast_3Castgraph_conv_38/Greater_3:z:0*

DstT0*

SrcT0
*=
_output_shapes+
):'???????????????????????????2
graph_conv_38/Cast_3?
graph_conv_38/transpose_3/permConst*
_output_shapes
:*
dtype0*!
valueB"          2 
graph_conv_38/transpose_3/perm?
graph_conv_38/transpose_3	Transposegraph_conv_38/Cast_3:y:0'graph_conv_38/transpose_3/perm:output:0*
T0*=
_output_shapes+
):'???????????????????????????2
graph_conv_38/transpose_3?
graph_conv_38/MatMul_7BatchMatMulV2graph_conv_38/transpose_3:y:0graph_conv_38/add_5:z:0*
T0*4
_output_shapes"
 :??????????????????2
graph_conv_38/MatMul_7}
graph_conv_38/Shape_9Shapegraph_conv_38/MatMul_7:output:0*
T0*
_output_shapes
:2
graph_conv_38/Shape_9?
%graph_conv_38/Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2'
%graph_conv_38/Sum_1/reduction_indices?
graph_conv_38/Sum_1Sumgraph_conv_38/Cast_3:y:0.graph_conv_38/Sum_1/reduction_indices:output:0*
T0*4
_output_shapes"
 :??????????????????*
	keep_dims(2
graph_conv_38/Sum_1s
graph_conv_38/add_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
graph_conv_38/add_7/y?
graph_conv_38/add_7AddV2graph_conv_38/Sum_1:output:0graph_conv_38/add_7/y:output:0*
T0*4
_output_shapes"
 :??????????????????2
graph_conv_38/add_7?
graph_conv_38/truediv_1RealDivgraph_conv_38/MatMul_7:output:0graph_conv_38/add_7:z:0*
T0*4
_output_shapes"
 :??????????????????2
graph_conv_38/truediv_1?
graph_conv_38/Relu_1Relugraph_conv_38/truediv_1:z:0*
T0*4
_output_shapes"
 :??????????????????2
graph_conv_38/Relu_1?
/tf.__operators__.getitem_25/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/tf.__operators__.getitem_25/strided_slice/stack?
1tf.__operators__.getitem_25/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1tf.__operators__.getitem_25/strided_slice/stack_1?
1tf.__operators__.getitem_25/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1tf.__operators__.getitem_25/strided_slice/stack_2?
)tf.__operators__.getitem_25/strided_sliceStridedSlice graph_conv_38/Relu:activations:08tf.__operators__.getitem_25/strided_slice/stack:output:0:tf.__operators__.getitem_25/strided_slice/stack_1:output:0:tf.__operators__.getitem_25/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2+
)tf.__operators__.getitem_25/strided_slice?
/tf.__operators__.getitem_24/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/tf.__operators__.getitem_24/strided_slice/stack?
1tf.__operators__.getitem_24/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1tf.__operators__.getitem_24/strided_slice/stack_1?
1tf.__operators__.getitem_24/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1tf.__operators__.getitem_24/strided_slice/stack_2?
)tf.__operators__.getitem_24/strided_sliceStridedSlice"graph_conv_38/Relu_1:activations:08tf.__operators__.getitem_24/strided_slice/stack:output:0:tf.__operators__.getitem_24/strided_slice/stack_1:output:0:tf.__operators__.getitem_24/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2+
)tf.__operators__.getitem_24/strided_slice?
"attention_12/MatMul/ReadVariableOpReadVariableOp+attention_12_matmul_readvariableop_resource*
_output_shapes

:*
dtype02$
"attention_12/MatMul/ReadVariableOp?
attention_12/MatMulMatMul2tf.__operators__.getitem_24/strided_slice:output:0*attention_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
attention_12/MatMul?
#attention_12/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2%
#attention_12/Mean/reduction_indices?
attention_12/MeanMeanattention_12/MatMul:product:0,attention_12/Mean/reduction_indices:output:0*
T0*
_output_shapes
:2
attention_12/Meano
attention_12/TanhTanhattention_12/Mean:output:0*
T0*
_output_shapes
:2
attention_12/Tanh?
attention_12/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ????2
attention_12/Reshape/shape?
attention_12/ReshapeReshapeattention_12/Tanh:y:0#attention_12/Reshape/shape:output:0*
T0*
_output_shapes

:2
attention_12/Reshape?
attention_12/transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2
attention_12/transpose/perm?
attention_12/transpose	Transposeattention_12/Reshape:output:0$attention_12/transpose/perm:output:0*
T0*
_output_shapes

:2
attention_12/transpose?
attention_12/MatMul_1MatMul2tf.__operators__.getitem_24/strided_slice:output:0attention_12/transpose:y:0*
T0*'
_output_shapes
:?????????2
attention_12/MatMul_1?
attention_12/SigmoidSigmoidattention_12/MatMul_1:product:0*
T0*'
_output_shapes
:?????????2
attention_12/Sigmoid?
attention_12/transpose_1/permConst*
_output_shapes
:*
dtype0*
valueB"       2
attention_12/transpose_1/perm?
attention_12/transpose_1	Transpose2tf.__operators__.getitem_24/strided_slice:output:0&attention_12/transpose_1/perm:output:0*
T0*'
_output_shapes
:?????????2
attention_12/transpose_1?
attention_12/MatMul_2MatMulattention_12/transpose_1:y:0attention_12/Sigmoid:y:0*
T0*
_output_shapes

:2
attention_12/MatMul_2?
attention_12/transpose_2/permConst*
_output_shapes
:*
dtype0*
valueB"       2
attention_12/transpose_2/perm?
attention_12/transpose_2	Transposeattention_12/MatMul_2:product:0&attention_12/transpose_2/perm:output:0*
T0*
_output_shapes

:2
attention_12/transpose_2?
$attention_12/MatMul_3/ReadVariableOpReadVariableOp+attention_12_matmul_readvariableop_resource*
_output_shapes

:*
dtype02&
$attention_12/MatMul_3/ReadVariableOp?
attention_12/MatMul_3MatMul2tf.__operators__.getitem_25/strided_slice:output:0,attention_12/MatMul_3/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
attention_12/MatMul_3?
%attention_12/Mean_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2'
%attention_12/Mean_1/reduction_indices?
attention_12/Mean_1Meanattention_12/MatMul_3:product:0.attention_12/Mean_1/reduction_indices:output:0*
T0*
_output_shapes
:2
attention_12/Mean_1u
attention_12/Tanh_1Tanhattention_12/Mean_1:output:0*
T0*
_output_shapes
:2
attention_12/Tanh_1?
attention_12/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"   ????2
attention_12/Reshape_1/shape?
attention_12/Reshape_1Reshapeattention_12/Tanh_1:y:0%attention_12/Reshape_1/shape:output:0*
T0*
_output_shapes

:2
attention_12/Reshape_1?
attention_12/transpose_3/permConst*
_output_shapes
:*
dtype0*
valueB"       2
attention_12/transpose_3/perm?
attention_12/transpose_3	Transposeattention_12/Reshape_1:output:0&attention_12/transpose_3/perm:output:0*
T0*
_output_shapes

:2
attention_12/transpose_3?
attention_12/MatMul_4MatMul2tf.__operators__.getitem_25/strided_slice:output:0attention_12/transpose_3:y:0*
T0*'
_output_shapes
:?????????2
attention_12/MatMul_4?
attention_12/Sigmoid_1Sigmoidattention_12/MatMul_4:product:0*
T0*'
_output_shapes
:?????????2
attention_12/Sigmoid_1?
attention_12/transpose_4/permConst*
_output_shapes
:*
dtype0*
valueB"       2
attention_12/transpose_4/perm?
attention_12/transpose_4	Transpose2tf.__operators__.getitem_25/strided_slice:output:0&attention_12/transpose_4/perm:output:0*
T0*'
_output_shapes
:?????????2
attention_12/transpose_4?
attention_12/MatMul_5MatMulattention_12/transpose_4:y:0attention_12/Sigmoid_1:y:0*
T0*
_output_shapes

:2
attention_12/MatMul_5?
attention_12/transpose_5/permConst*
_output_shapes
:*
dtype0*
valueB"       2
attention_12/transpose_5/perm?
attention_12/transpose_5	Transposeattention_12/MatMul_5:product:0&attention_12/transpose_5/perm:output:0*
T0*
_output_shapes

:2
attention_12/transpose_5?
neural_tensor_layer_12/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2
neural_tensor_layer_12/Shape?
*neural_tensor_layer_12/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*neural_tensor_layer_12/strided_slice/stack?
,neural_tensor_layer_12/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,neural_tensor_layer_12/strided_slice/stack_1?
,neural_tensor_layer_12/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,neural_tensor_layer_12/strided_slice/stack_2?
$neural_tensor_layer_12/strided_sliceStridedSlice%neural_tensor_layer_12/Shape:output:03neural_tensor_layer_12/strided_slice/stack:output:05neural_tensor_layer_12/strided_slice/stack_1:output:05neural_tensor_layer_12/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$neural_tensor_layer_12/strided_slice?
"neural_tensor_layer_12/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2$
"neural_tensor_layer_12/concat/axis?
neural_tensor_layer_12/concatConcatV2attention_12/transpose_2:y:0attention_12/transpose_5:y:0+neural_tensor_layer_12/concat/axis:output:0*
N*
T0*
_output_shapes

: 2
neural_tensor_layer_12/concat?
,neural_tensor_layer_12/MatMul/ReadVariableOpReadVariableOp5neural_tensor_layer_12_matmul_readvariableop_resource*
_output_shapes

: *
dtype02.
,neural_tensor_layer_12/MatMul/ReadVariableOp?
neural_tensor_layer_12/MatMulMatMul&neural_tensor_layer_12/concat:output:04neural_tensor_layer_12/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
neural_tensor_layer_12/MatMul?
%neural_tensor_layer_12/ReadVariableOpReadVariableOp.neural_tensor_layer_12_readvariableop_resource*"
_output_shapes
:*
dtype02'
%neural_tensor_layer_12/ReadVariableOp?
,neural_tensor_layer_12/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,neural_tensor_layer_12/strided_slice_1/stack?
.neural_tensor_layer_12/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.neural_tensor_layer_12/strided_slice_1/stack_1?
.neural_tensor_layer_12/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.neural_tensor_layer_12/strided_slice_1/stack_2?
&neural_tensor_layer_12/strided_slice_1StridedSlice-neural_tensor_layer_12/ReadVariableOp:value:05neural_tensor_layer_12/strided_slice_1/stack:output:07neural_tensor_layer_12/strided_slice_1/stack_1:output:07neural_tensor_layer_12/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2(
&neural_tensor_layer_12/strided_slice_1?
neural_tensor_layer_12/MatMul_1MatMulattention_12/transpose_2:y:0/neural_tensor_layer_12/strided_slice_1:output:0*
T0*
_output_shapes

:2!
neural_tensor_layer_12/MatMul_1?
neural_tensor_layer_12/mulMulattention_12/transpose_5:y:0)neural_tensor_layer_12/MatMul_1:product:0*
T0*
_output_shapes

:2
neural_tensor_layer_12/mul?
)neural_tensor_layer_12/add/ReadVariableOpReadVariableOp2neural_tensor_layer_12_add_readvariableop_resource*
_output_shapes
:*
dtype02+
)neural_tensor_layer_12/add/ReadVariableOp?
neural_tensor_layer_12/addAddV2neural_tensor_layer_12/mul:z:01neural_tensor_layer_12/add/ReadVariableOp:value:0*
T0*
_output_shapes

:2
neural_tensor_layer_12/add?
,neural_tensor_layer_12/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2.
,neural_tensor_layer_12/Sum/reduction_indices?
neural_tensor_layer_12/SumSumneural_tensor_layer_12/add:z:05neural_tensor_layer_12/Sum/reduction_indices:output:0*
T0*
_output_shapes
:2
neural_tensor_layer_12/Sum?
'neural_tensor_layer_12/ReadVariableOp_1ReadVariableOp.neural_tensor_layer_12_readvariableop_resource*"
_output_shapes
:*
dtype02)
'neural_tensor_layer_12/ReadVariableOp_1?
,neural_tensor_layer_12/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2.
,neural_tensor_layer_12/strided_slice_2/stack?
.neural_tensor_layer_12/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.neural_tensor_layer_12/strided_slice_2/stack_1?
.neural_tensor_layer_12/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.neural_tensor_layer_12/strided_slice_2/stack_2?
&neural_tensor_layer_12/strided_slice_2StridedSlice/neural_tensor_layer_12/ReadVariableOp_1:value:05neural_tensor_layer_12/strided_slice_2/stack:output:07neural_tensor_layer_12/strided_slice_2/stack_1:output:07neural_tensor_layer_12/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2(
&neural_tensor_layer_12/strided_slice_2?
neural_tensor_layer_12/MatMul_2MatMulattention_12/transpose_2:y:0/neural_tensor_layer_12/strided_slice_2:output:0*
T0*
_output_shapes

:2!
neural_tensor_layer_12/MatMul_2?
neural_tensor_layer_12/mul_1Mulattention_12/transpose_5:y:0)neural_tensor_layer_12/MatMul_2:product:0*
T0*
_output_shapes

:2
neural_tensor_layer_12/mul_1?
+neural_tensor_layer_12/add_1/ReadVariableOpReadVariableOp2neural_tensor_layer_12_add_readvariableop_resource*
_output_shapes
:*
dtype02-
+neural_tensor_layer_12/add_1/ReadVariableOp?
neural_tensor_layer_12/add_1AddV2 neural_tensor_layer_12/mul_1:z:03neural_tensor_layer_12/add_1/ReadVariableOp:value:0*
T0*
_output_shapes

:2
neural_tensor_layer_12/add_1?
.neural_tensor_layer_12/Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :20
.neural_tensor_layer_12/Sum_1/reduction_indices?
neural_tensor_layer_12/Sum_1Sum neural_tensor_layer_12/add_1:z:07neural_tensor_layer_12/Sum_1/reduction_indices:output:0*
T0*
_output_shapes
:2
neural_tensor_layer_12/Sum_1?
'neural_tensor_layer_12/ReadVariableOp_2ReadVariableOp.neural_tensor_layer_12_readvariableop_resource*"
_output_shapes
:*
dtype02)
'neural_tensor_layer_12/ReadVariableOp_2?
,neural_tensor_layer_12/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:2.
,neural_tensor_layer_12/strided_slice_3/stack?
.neural_tensor_layer_12/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.neural_tensor_layer_12/strided_slice_3/stack_1?
.neural_tensor_layer_12/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.neural_tensor_layer_12/strided_slice_3/stack_2?
&neural_tensor_layer_12/strided_slice_3StridedSlice/neural_tensor_layer_12/ReadVariableOp_2:value:05neural_tensor_layer_12/strided_slice_3/stack:output:07neural_tensor_layer_12/strided_slice_3/stack_1:output:07neural_tensor_layer_12/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2(
&neural_tensor_layer_12/strided_slice_3?
neural_tensor_layer_12/MatMul_3MatMulattention_12/transpose_2:y:0/neural_tensor_layer_12/strided_slice_3:output:0*
T0*
_output_shapes

:2!
neural_tensor_layer_12/MatMul_3?
neural_tensor_layer_12/mul_2Mulattention_12/transpose_5:y:0)neural_tensor_layer_12/MatMul_3:product:0*
T0*
_output_shapes

:2
neural_tensor_layer_12/mul_2?
+neural_tensor_layer_12/add_2/ReadVariableOpReadVariableOp2neural_tensor_layer_12_add_readvariableop_resource*
_output_shapes
:*
dtype02-
+neural_tensor_layer_12/add_2/ReadVariableOp?
neural_tensor_layer_12/add_2AddV2 neural_tensor_layer_12/mul_2:z:03neural_tensor_layer_12/add_2/ReadVariableOp:value:0*
T0*
_output_shapes

:2
neural_tensor_layer_12/add_2?
.neural_tensor_layer_12/Sum_2/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :20
.neural_tensor_layer_12/Sum_2/reduction_indices?
neural_tensor_layer_12/Sum_2Sum neural_tensor_layer_12/add_2:z:07neural_tensor_layer_12/Sum_2/reduction_indices:output:0*
T0*
_output_shapes
:2
neural_tensor_layer_12/Sum_2?
'neural_tensor_layer_12/ReadVariableOp_3ReadVariableOp.neural_tensor_layer_12_readvariableop_resource*"
_output_shapes
:*
dtype02)
'neural_tensor_layer_12/ReadVariableOp_3?
,neural_tensor_layer_12/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:2.
,neural_tensor_layer_12/strided_slice_4/stack?
.neural_tensor_layer_12/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.neural_tensor_layer_12/strided_slice_4/stack_1?
.neural_tensor_layer_12/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.neural_tensor_layer_12/strided_slice_4/stack_2?
&neural_tensor_layer_12/strided_slice_4StridedSlice/neural_tensor_layer_12/ReadVariableOp_3:value:05neural_tensor_layer_12/strided_slice_4/stack:output:07neural_tensor_layer_12/strided_slice_4/stack_1:output:07neural_tensor_layer_12/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2(
&neural_tensor_layer_12/strided_slice_4?
neural_tensor_layer_12/MatMul_4MatMulattention_12/transpose_2:y:0/neural_tensor_layer_12/strided_slice_4:output:0*
T0*
_output_shapes

:2!
neural_tensor_layer_12/MatMul_4?
neural_tensor_layer_12/mul_3Mulattention_12/transpose_5:y:0)neural_tensor_layer_12/MatMul_4:product:0*
T0*
_output_shapes

:2
neural_tensor_layer_12/mul_3?
+neural_tensor_layer_12/add_3/ReadVariableOpReadVariableOp2neural_tensor_layer_12_add_readvariableop_resource*
_output_shapes
:*
dtype02-
+neural_tensor_layer_12/add_3/ReadVariableOp?
neural_tensor_layer_12/add_3AddV2 neural_tensor_layer_12/mul_3:z:03neural_tensor_layer_12/add_3/ReadVariableOp:value:0*
T0*
_output_shapes

:2
neural_tensor_layer_12/add_3?
.neural_tensor_layer_12/Sum_3/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :20
.neural_tensor_layer_12/Sum_3/reduction_indices?
neural_tensor_layer_12/Sum_3Sum neural_tensor_layer_12/add_3:z:07neural_tensor_layer_12/Sum_3/reduction_indices:output:0*
T0*
_output_shapes
:2
neural_tensor_layer_12/Sum_3?
'neural_tensor_layer_12/ReadVariableOp_4ReadVariableOp.neural_tensor_layer_12_readvariableop_resource*"
_output_shapes
:*
dtype02)
'neural_tensor_layer_12/ReadVariableOp_4?
,neural_tensor_layer_12/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:2.
,neural_tensor_layer_12/strided_slice_5/stack?
.neural_tensor_layer_12/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.neural_tensor_layer_12/strided_slice_5/stack_1?
.neural_tensor_layer_12/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.neural_tensor_layer_12/strided_slice_5/stack_2?
&neural_tensor_layer_12/strided_slice_5StridedSlice/neural_tensor_layer_12/ReadVariableOp_4:value:05neural_tensor_layer_12/strided_slice_5/stack:output:07neural_tensor_layer_12/strided_slice_5/stack_1:output:07neural_tensor_layer_12/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2(
&neural_tensor_layer_12/strided_slice_5?
neural_tensor_layer_12/MatMul_5MatMulattention_12/transpose_2:y:0/neural_tensor_layer_12/strided_slice_5:output:0*
T0*
_output_shapes

:2!
neural_tensor_layer_12/MatMul_5?
neural_tensor_layer_12/mul_4Mulattention_12/transpose_5:y:0)neural_tensor_layer_12/MatMul_5:product:0*
T0*
_output_shapes

:2
neural_tensor_layer_12/mul_4?
+neural_tensor_layer_12/add_4/ReadVariableOpReadVariableOp2neural_tensor_layer_12_add_readvariableop_resource*
_output_shapes
:*
dtype02-
+neural_tensor_layer_12/add_4/ReadVariableOp?
neural_tensor_layer_12/add_4AddV2 neural_tensor_layer_12/mul_4:z:03neural_tensor_layer_12/add_4/ReadVariableOp:value:0*
T0*
_output_shapes

:2
neural_tensor_layer_12/add_4?
.neural_tensor_layer_12/Sum_4/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :20
.neural_tensor_layer_12/Sum_4/reduction_indices?
neural_tensor_layer_12/Sum_4Sum neural_tensor_layer_12/add_4:z:07neural_tensor_layer_12/Sum_4/reduction_indices:output:0*
T0*
_output_shapes
:2
neural_tensor_layer_12/Sum_4?
'neural_tensor_layer_12/ReadVariableOp_5ReadVariableOp.neural_tensor_layer_12_readvariableop_resource*"
_output_shapes
:*
dtype02)
'neural_tensor_layer_12/ReadVariableOp_5?
,neural_tensor_layer_12/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB:2.
,neural_tensor_layer_12/strided_slice_6/stack?
.neural_tensor_layer_12/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.neural_tensor_layer_12/strided_slice_6/stack_1?
.neural_tensor_layer_12/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.neural_tensor_layer_12/strided_slice_6/stack_2?
&neural_tensor_layer_12/strided_slice_6StridedSlice/neural_tensor_layer_12/ReadVariableOp_5:value:05neural_tensor_layer_12/strided_slice_6/stack:output:07neural_tensor_layer_12/strided_slice_6/stack_1:output:07neural_tensor_layer_12/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2(
&neural_tensor_layer_12/strided_slice_6?
neural_tensor_layer_12/MatMul_6MatMulattention_12/transpose_2:y:0/neural_tensor_layer_12/strided_slice_6:output:0*
T0*
_output_shapes

:2!
neural_tensor_layer_12/MatMul_6?
neural_tensor_layer_12/mul_5Mulattention_12/transpose_5:y:0)neural_tensor_layer_12/MatMul_6:product:0*
T0*
_output_shapes

:2
neural_tensor_layer_12/mul_5?
+neural_tensor_layer_12/add_5/ReadVariableOpReadVariableOp2neural_tensor_layer_12_add_readvariableop_resource*
_output_shapes
:*
dtype02-
+neural_tensor_layer_12/add_5/ReadVariableOp?
neural_tensor_layer_12/add_5AddV2 neural_tensor_layer_12/mul_5:z:03neural_tensor_layer_12/add_5/ReadVariableOp:value:0*
T0*
_output_shapes

:2
neural_tensor_layer_12/add_5?
.neural_tensor_layer_12/Sum_5/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :20
.neural_tensor_layer_12/Sum_5/reduction_indices?
neural_tensor_layer_12/Sum_5Sum neural_tensor_layer_12/add_5:z:07neural_tensor_layer_12/Sum_5/reduction_indices:output:0*
T0*
_output_shapes
:2
neural_tensor_layer_12/Sum_5?
'neural_tensor_layer_12/ReadVariableOp_6ReadVariableOp.neural_tensor_layer_12_readvariableop_resource*"
_output_shapes
:*
dtype02)
'neural_tensor_layer_12/ReadVariableOp_6?
,neural_tensor_layer_12/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB:2.
,neural_tensor_layer_12/strided_slice_7/stack?
.neural_tensor_layer_12/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.neural_tensor_layer_12/strided_slice_7/stack_1?
.neural_tensor_layer_12/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.neural_tensor_layer_12/strided_slice_7/stack_2?
&neural_tensor_layer_12/strided_slice_7StridedSlice/neural_tensor_layer_12/ReadVariableOp_6:value:05neural_tensor_layer_12/strided_slice_7/stack:output:07neural_tensor_layer_12/strided_slice_7/stack_1:output:07neural_tensor_layer_12/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2(
&neural_tensor_layer_12/strided_slice_7?
neural_tensor_layer_12/MatMul_7MatMulattention_12/transpose_2:y:0/neural_tensor_layer_12/strided_slice_7:output:0*
T0*
_output_shapes

:2!
neural_tensor_layer_12/MatMul_7?
neural_tensor_layer_12/mul_6Mulattention_12/transpose_5:y:0)neural_tensor_layer_12/MatMul_7:product:0*
T0*
_output_shapes

:2
neural_tensor_layer_12/mul_6?
+neural_tensor_layer_12/add_6/ReadVariableOpReadVariableOp2neural_tensor_layer_12_add_readvariableop_resource*
_output_shapes
:*
dtype02-
+neural_tensor_layer_12/add_6/ReadVariableOp?
neural_tensor_layer_12/add_6AddV2 neural_tensor_layer_12/mul_6:z:03neural_tensor_layer_12/add_6/ReadVariableOp:value:0*
T0*
_output_shapes

:2
neural_tensor_layer_12/add_6?
.neural_tensor_layer_12/Sum_6/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :20
.neural_tensor_layer_12/Sum_6/reduction_indices?
neural_tensor_layer_12/Sum_6Sum neural_tensor_layer_12/add_6:z:07neural_tensor_layer_12/Sum_6/reduction_indices:output:0*
T0*
_output_shapes
:2
neural_tensor_layer_12/Sum_6?
'neural_tensor_layer_12/ReadVariableOp_7ReadVariableOp.neural_tensor_layer_12_readvariableop_resource*"
_output_shapes
:*
dtype02)
'neural_tensor_layer_12/ReadVariableOp_7?
,neural_tensor_layer_12/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB:2.
,neural_tensor_layer_12/strided_slice_8/stack?
.neural_tensor_layer_12/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.neural_tensor_layer_12/strided_slice_8/stack_1?
.neural_tensor_layer_12/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.neural_tensor_layer_12/strided_slice_8/stack_2?
&neural_tensor_layer_12/strided_slice_8StridedSlice/neural_tensor_layer_12/ReadVariableOp_7:value:05neural_tensor_layer_12/strided_slice_8/stack:output:07neural_tensor_layer_12/strided_slice_8/stack_1:output:07neural_tensor_layer_12/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2(
&neural_tensor_layer_12/strided_slice_8?
neural_tensor_layer_12/MatMul_8MatMulattention_12/transpose_2:y:0/neural_tensor_layer_12/strided_slice_8:output:0*
T0*
_output_shapes

:2!
neural_tensor_layer_12/MatMul_8?
neural_tensor_layer_12/mul_7Mulattention_12/transpose_5:y:0)neural_tensor_layer_12/MatMul_8:product:0*
T0*
_output_shapes

:2
neural_tensor_layer_12/mul_7?
+neural_tensor_layer_12/add_7/ReadVariableOpReadVariableOp2neural_tensor_layer_12_add_readvariableop_resource*
_output_shapes
:*
dtype02-
+neural_tensor_layer_12/add_7/ReadVariableOp?
neural_tensor_layer_12/add_7AddV2 neural_tensor_layer_12/mul_7:z:03neural_tensor_layer_12/add_7/ReadVariableOp:value:0*
T0*
_output_shapes

:2
neural_tensor_layer_12/add_7?
.neural_tensor_layer_12/Sum_7/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :20
.neural_tensor_layer_12/Sum_7/reduction_indices?
neural_tensor_layer_12/Sum_7Sum neural_tensor_layer_12/add_7:z:07neural_tensor_layer_12/Sum_7/reduction_indices:output:0*
T0*
_output_shapes
:2
neural_tensor_layer_12/Sum_7?
'neural_tensor_layer_12/ReadVariableOp_8ReadVariableOp.neural_tensor_layer_12_readvariableop_resource*"
_output_shapes
:*
dtype02)
'neural_tensor_layer_12/ReadVariableOp_8?
,neural_tensor_layer_12/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB:2.
,neural_tensor_layer_12/strided_slice_9/stack?
.neural_tensor_layer_12/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:	20
.neural_tensor_layer_12/strided_slice_9/stack_1?
.neural_tensor_layer_12/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.neural_tensor_layer_12/strided_slice_9/stack_2?
&neural_tensor_layer_12/strided_slice_9StridedSlice/neural_tensor_layer_12/ReadVariableOp_8:value:05neural_tensor_layer_12/strided_slice_9/stack:output:07neural_tensor_layer_12/strided_slice_9/stack_1:output:07neural_tensor_layer_12/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2(
&neural_tensor_layer_12/strided_slice_9?
neural_tensor_layer_12/MatMul_9MatMulattention_12/transpose_2:y:0/neural_tensor_layer_12/strided_slice_9:output:0*
T0*
_output_shapes

:2!
neural_tensor_layer_12/MatMul_9?
neural_tensor_layer_12/mul_8Mulattention_12/transpose_5:y:0)neural_tensor_layer_12/MatMul_9:product:0*
T0*
_output_shapes

:2
neural_tensor_layer_12/mul_8?
+neural_tensor_layer_12/add_8/ReadVariableOpReadVariableOp2neural_tensor_layer_12_add_readvariableop_resource*
_output_shapes
:*
dtype02-
+neural_tensor_layer_12/add_8/ReadVariableOp?
neural_tensor_layer_12/add_8AddV2 neural_tensor_layer_12/mul_8:z:03neural_tensor_layer_12/add_8/ReadVariableOp:value:0*
T0*
_output_shapes

:2
neural_tensor_layer_12/add_8?
.neural_tensor_layer_12/Sum_8/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :20
.neural_tensor_layer_12/Sum_8/reduction_indices?
neural_tensor_layer_12/Sum_8Sum neural_tensor_layer_12/add_8:z:07neural_tensor_layer_12/Sum_8/reduction_indices:output:0*
T0*
_output_shapes
:2
neural_tensor_layer_12/Sum_8?
'neural_tensor_layer_12/ReadVariableOp_9ReadVariableOp.neural_tensor_layer_12_readvariableop_resource*"
_output_shapes
:*
dtype02)
'neural_tensor_layer_12/ReadVariableOp_9?
-neural_tensor_layer_12/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB:	2/
-neural_tensor_layer_12/strided_slice_10/stack?
/neural_tensor_layer_12/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
21
/neural_tensor_layer_12/strided_slice_10/stack_1?
/neural_tensor_layer_12/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/neural_tensor_layer_12/strided_slice_10/stack_2?
'neural_tensor_layer_12/strided_slice_10StridedSlice/neural_tensor_layer_12/ReadVariableOp_9:value:06neural_tensor_layer_12/strided_slice_10/stack:output:08neural_tensor_layer_12/strided_slice_10/stack_1:output:08neural_tensor_layer_12/strided_slice_10/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2)
'neural_tensor_layer_12/strided_slice_10?
 neural_tensor_layer_12/MatMul_10MatMulattention_12/transpose_2:y:00neural_tensor_layer_12/strided_slice_10:output:0*
T0*
_output_shapes

:2"
 neural_tensor_layer_12/MatMul_10?
neural_tensor_layer_12/mul_9Mulattention_12/transpose_5:y:0*neural_tensor_layer_12/MatMul_10:product:0*
T0*
_output_shapes

:2
neural_tensor_layer_12/mul_9?
+neural_tensor_layer_12/add_9/ReadVariableOpReadVariableOp2neural_tensor_layer_12_add_readvariableop_resource*
_output_shapes
:*
dtype02-
+neural_tensor_layer_12/add_9/ReadVariableOp?
neural_tensor_layer_12/add_9AddV2 neural_tensor_layer_12/mul_9:z:03neural_tensor_layer_12/add_9/ReadVariableOp:value:0*
T0*
_output_shapes

:2
neural_tensor_layer_12/add_9?
.neural_tensor_layer_12/Sum_9/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :20
.neural_tensor_layer_12/Sum_9/reduction_indices?
neural_tensor_layer_12/Sum_9Sum neural_tensor_layer_12/add_9:z:07neural_tensor_layer_12/Sum_9/reduction_indices:output:0*
T0*
_output_shapes
:2
neural_tensor_layer_12/Sum_9?
(neural_tensor_layer_12/ReadVariableOp_10ReadVariableOp.neural_tensor_layer_12_readvariableop_resource*"
_output_shapes
:*
dtype02*
(neural_tensor_layer_12/ReadVariableOp_10?
-neural_tensor_layer_12/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:
2/
-neural_tensor_layer_12/strided_slice_11/stack?
/neural_tensor_layer_12/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/neural_tensor_layer_12/strided_slice_11/stack_1?
/neural_tensor_layer_12/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/neural_tensor_layer_12/strided_slice_11/stack_2?
'neural_tensor_layer_12/strided_slice_11StridedSlice0neural_tensor_layer_12/ReadVariableOp_10:value:06neural_tensor_layer_12/strided_slice_11/stack:output:08neural_tensor_layer_12/strided_slice_11/stack_1:output:08neural_tensor_layer_12/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2)
'neural_tensor_layer_12/strided_slice_11?
 neural_tensor_layer_12/MatMul_11MatMulattention_12/transpose_2:y:00neural_tensor_layer_12/strided_slice_11:output:0*
T0*
_output_shapes

:2"
 neural_tensor_layer_12/MatMul_11?
neural_tensor_layer_12/mul_10Mulattention_12/transpose_5:y:0*neural_tensor_layer_12/MatMul_11:product:0*
T0*
_output_shapes

:2
neural_tensor_layer_12/mul_10?
,neural_tensor_layer_12/add_10/ReadVariableOpReadVariableOp2neural_tensor_layer_12_add_readvariableop_resource*
_output_shapes
:*
dtype02.
,neural_tensor_layer_12/add_10/ReadVariableOp?
neural_tensor_layer_12/add_10AddV2!neural_tensor_layer_12/mul_10:z:04neural_tensor_layer_12/add_10/ReadVariableOp:value:0*
T0*
_output_shapes

:2
neural_tensor_layer_12/add_10?
/neural_tensor_layer_12/Sum_10/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :21
/neural_tensor_layer_12/Sum_10/reduction_indices?
neural_tensor_layer_12/Sum_10Sum!neural_tensor_layer_12/add_10:z:08neural_tensor_layer_12/Sum_10/reduction_indices:output:0*
T0*
_output_shapes
:2
neural_tensor_layer_12/Sum_10?
(neural_tensor_layer_12/ReadVariableOp_11ReadVariableOp.neural_tensor_layer_12_readvariableop_resource*"
_output_shapes
:*
dtype02*
(neural_tensor_layer_12/ReadVariableOp_11?
-neural_tensor_layer_12/strided_slice_12/stackConst*
_output_shapes
:*
dtype0*
valueB:2/
-neural_tensor_layer_12/strided_slice_12/stack?
/neural_tensor_layer_12/strided_slice_12/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/neural_tensor_layer_12/strided_slice_12/stack_1?
/neural_tensor_layer_12/strided_slice_12/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/neural_tensor_layer_12/strided_slice_12/stack_2?
'neural_tensor_layer_12/strided_slice_12StridedSlice0neural_tensor_layer_12/ReadVariableOp_11:value:06neural_tensor_layer_12/strided_slice_12/stack:output:08neural_tensor_layer_12/strided_slice_12/stack_1:output:08neural_tensor_layer_12/strided_slice_12/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2)
'neural_tensor_layer_12/strided_slice_12?
 neural_tensor_layer_12/MatMul_12MatMulattention_12/transpose_2:y:00neural_tensor_layer_12/strided_slice_12:output:0*
T0*
_output_shapes

:2"
 neural_tensor_layer_12/MatMul_12?
neural_tensor_layer_12/mul_11Mulattention_12/transpose_5:y:0*neural_tensor_layer_12/MatMul_12:product:0*
T0*
_output_shapes

:2
neural_tensor_layer_12/mul_11?
,neural_tensor_layer_12/add_11/ReadVariableOpReadVariableOp2neural_tensor_layer_12_add_readvariableop_resource*
_output_shapes
:*
dtype02.
,neural_tensor_layer_12/add_11/ReadVariableOp?
neural_tensor_layer_12/add_11AddV2!neural_tensor_layer_12/mul_11:z:04neural_tensor_layer_12/add_11/ReadVariableOp:value:0*
T0*
_output_shapes

:2
neural_tensor_layer_12/add_11?
/neural_tensor_layer_12/Sum_11/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :21
/neural_tensor_layer_12/Sum_11/reduction_indices?
neural_tensor_layer_12/Sum_11Sum!neural_tensor_layer_12/add_11:z:08neural_tensor_layer_12/Sum_11/reduction_indices:output:0*
T0*
_output_shapes
:2
neural_tensor_layer_12/Sum_11?
(neural_tensor_layer_12/ReadVariableOp_12ReadVariableOp.neural_tensor_layer_12_readvariableop_resource*"
_output_shapes
:*
dtype02*
(neural_tensor_layer_12/ReadVariableOp_12?
-neural_tensor_layer_12/strided_slice_13/stackConst*
_output_shapes
:*
dtype0*
valueB:2/
-neural_tensor_layer_12/strided_slice_13/stack?
/neural_tensor_layer_12/strided_slice_13/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/neural_tensor_layer_12/strided_slice_13/stack_1?
/neural_tensor_layer_12/strided_slice_13/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/neural_tensor_layer_12/strided_slice_13/stack_2?
'neural_tensor_layer_12/strided_slice_13StridedSlice0neural_tensor_layer_12/ReadVariableOp_12:value:06neural_tensor_layer_12/strided_slice_13/stack:output:08neural_tensor_layer_12/strided_slice_13/stack_1:output:08neural_tensor_layer_12/strided_slice_13/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2)
'neural_tensor_layer_12/strided_slice_13?
 neural_tensor_layer_12/MatMul_13MatMulattention_12/transpose_2:y:00neural_tensor_layer_12/strided_slice_13:output:0*
T0*
_output_shapes

:2"
 neural_tensor_layer_12/MatMul_13?
neural_tensor_layer_12/mul_12Mulattention_12/transpose_5:y:0*neural_tensor_layer_12/MatMul_13:product:0*
T0*
_output_shapes

:2
neural_tensor_layer_12/mul_12?
,neural_tensor_layer_12/add_12/ReadVariableOpReadVariableOp2neural_tensor_layer_12_add_readvariableop_resource*
_output_shapes
:*
dtype02.
,neural_tensor_layer_12/add_12/ReadVariableOp?
neural_tensor_layer_12/add_12AddV2!neural_tensor_layer_12/mul_12:z:04neural_tensor_layer_12/add_12/ReadVariableOp:value:0*
T0*
_output_shapes

:2
neural_tensor_layer_12/add_12?
/neural_tensor_layer_12/Sum_12/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :21
/neural_tensor_layer_12/Sum_12/reduction_indices?
neural_tensor_layer_12/Sum_12Sum!neural_tensor_layer_12/add_12:z:08neural_tensor_layer_12/Sum_12/reduction_indices:output:0*
T0*
_output_shapes
:2
neural_tensor_layer_12/Sum_12?
(neural_tensor_layer_12/ReadVariableOp_13ReadVariableOp.neural_tensor_layer_12_readvariableop_resource*"
_output_shapes
:*
dtype02*
(neural_tensor_layer_12/ReadVariableOp_13?
-neural_tensor_layer_12/strided_slice_14/stackConst*
_output_shapes
:*
dtype0*
valueB:2/
-neural_tensor_layer_12/strided_slice_14/stack?
/neural_tensor_layer_12/strided_slice_14/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/neural_tensor_layer_12/strided_slice_14/stack_1?
/neural_tensor_layer_12/strided_slice_14/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/neural_tensor_layer_12/strided_slice_14/stack_2?
'neural_tensor_layer_12/strided_slice_14StridedSlice0neural_tensor_layer_12/ReadVariableOp_13:value:06neural_tensor_layer_12/strided_slice_14/stack:output:08neural_tensor_layer_12/strided_slice_14/stack_1:output:08neural_tensor_layer_12/strided_slice_14/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2)
'neural_tensor_layer_12/strided_slice_14?
 neural_tensor_layer_12/MatMul_14MatMulattention_12/transpose_2:y:00neural_tensor_layer_12/strided_slice_14:output:0*
T0*
_output_shapes

:2"
 neural_tensor_layer_12/MatMul_14?
neural_tensor_layer_12/mul_13Mulattention_12/transpose_5:y:0*neural_tensor_layer_12/MatMul_14:product:0*
T0*
_output_shapes

:2
neural_tensor_layer_12/mul_13?
,neural_tensor_layer_12/add_13/ReadVariableOpReadVariableOp2neural_tensor_layer_12_add_readvariableop_resource*
_output_shapes
:*
dtype02.
,neural_tensor_layer_12/add_13/ReadVariableOp?
neural_tensor_layer_12/add_13AddV2!neural_tensor_layer_12/mul_13:z:04neural_tensor_layer_12/add_13/ReadVariableOp:value:0*
T0*
_output_shapes

:2
neural_tensor_layer_12/add_13?
/neural_tensor_layer_12/Sum_13/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :21
/neural_tensor_layer_12/Sum_13/reduction_indices?
neural_tensor_layer_12/Sum_13Sum!neural_tensor_layer_12/add_13:z:08neural_tensor_layer_12/Sum_13/reduction_indices:output:0*
T0*
_output_shapes
:2
neural_tensor_layer_12/Sum_13?
(neural_tensor_layer_12/ReadVariableOp_14ReadVariableOp.neural_tensor_layer_12_readvariableop_resource*"
_output_shapes
:*
dtype02*
(neural_tensor_layer_12/ReadVariableOp_14?
-neural_tensor_layer_12/strided_slice_15/stackConst*
_output_shapes
:*
dtype0*
valueB:2/
-neural_tensor_layer_12/strided_slice_15/stack?
/neural_tensor_layer_12/strided_slice_15/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/neural_tensor_layer_12/strided_slice_15/stack_1?
/neural_tensor_layer_12/strided_slice_15/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/neural_tensor_layer_12/strided_slice_15/stack_2?
'neural_tensor_layer_12/strided_slice_15StridedSlice0neural_tensor_layer_12/ReadVariableOp_14:value:06neural_tensor_layer_12/strided_slice_15/stack:output:08neural_tensor_layer_12/strided_slice_15/stack_1:output:08neural_tensor_layer_12/strided_slice_15/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2)
'neural_tensor_layer_12/strided_slice_15?
 neural_tensor_layer_12/MatMul_15MatMulattention_12/transpose_2:y:00neural_tensor_layer_12/strided_slice_15:output:0*
T0*
_output_shapes

:2"
 neural_tensor_layer_12/MatMul_15?
neural_tensor_layer_12/mul_14Mulattention_12/transpose_5:y:0*neural_tensor_layer_12/MatMul_15:product:0*
T0*
_output_shapes

:2
neural_tensor_layer_12/mul_14?
,neural_tensor_layer_12/add_14/ReadVariableOpReadVariableOp2neural_tensor_layer_12_add_readvariableop_resource*
_output_shapes
:*
dtype02.
,neural_tensor_layer_12/add_14/ReadVariableOp?
neural_tensor_layer_12/add_14AddV2!neural_tensor_layer_12/mul_14:z:04neural_tensor_layer_12/add_14/ReadVariableOp:value:0*
T0*
_output_shapes

:2
neural_tensor_layer_12/add_14?
/neural_tensor_layer_12/Sum_14/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :21
/neural_tensor_layer_12/Sum_14/reduction_indices?
neural_tensor_layer_12/Sum_14Sum!neural_tensor_layer_12/add_14:z:08neural_tensor_layer_12/Sum_14/reduction_indices:output:0*
T0*
_output_shapes
:2
neural_tensor_layer_12/Sum_14?
(neural_tensor_layer_12/ReadVariableOp_15ReadVariableOp.neural_tensor_layer_12_readvariableop_resource*"
_output_shapes
:*
dtype02*
(neural_tensor_layer_12/ReadVariableOp_15?
-neural_tensor_layer_12/strided_slice_16/stackConst*
_output_shapes
:*
dtype0*
valueB:2/
-neural_tensor_layer_12/strided_slice_16/stack?
/neural_tensor_layer_12/strided_slice_16/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/neural_tensor_layer_12/strided_slice_16/stack_1?
/neural_tensor_layer_12/strided_slice_16/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/neural_tensor_layer_12/strided_slice_16/stack_2?
'neural_tensor_layer_12/strided_slice_16StridedSlice0neural_tensor_layer_12/ReadVariableOp_15:value:06neural_tensor_layer_12/strided_slice_16/stack:output:08neural_tensor_layer_12/strided_slice_16/stack_1:output:08neural_tensor_layer_12/strided_slice_16/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2)
'neural_tensor_layer_12/strided_slice_16?
 neural_tensor_layer_12/MatMul_16MatMulattention_12/transpose_2:y:00neural_tensor_layer_12/strided_slice_16:output:0*
T0*
_output_shapes

:2"
 neural_tensor_layer_12/MatMul_16?
neural_tensor_layer_12/mul_15Mulattention_12/transpose_5:y:0*neural_tensor_layer_12/MatMul_16:product:0*
T0*
_output_shapes

:2
neural_tensor_layer_12/mul_15?
,neural_tensor_layer_12/add_15/ReadVariableOpReadVariableOp2neural_tensor_layer_12_add_readvariableop_resource*
_output_shapes
:*
dtype02.
,neural_tensor_layer_12/add_15/ReadVariableOp?
neural_tensor_layer_12/add_15AddV2!neural_tensor_layer_12/mul_15:z:04neural_tensor_layer_12/add_15/ReadVariableOp:value:0*
T0*
_output_shapes

:2
neural_tensor_layer_12/add_15?
/neural_tensor_layer_12/Sum_15/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :21
/neural_tensor_layer_12/Sum_15/reduction_indices?
neural_tensor_layer_12/Sum_15Sum!neural_tensor_layer_12/add_15:z:08neural_tensor_layer_12/Sum_15/reduction_indices:output:0*
T0*
_output_shapes
:2
neural_tensor_layer_12/Sum_15?
$neural_tensor_layer_12/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2&
$neural_tensor_layer_12/concat_1/axis?
neural_tensor_layer_12/concat_1ConcatV2#neural_tensor_layer_12/Sum:output:0%neural_tensor_layer_12/Sum_1:output:0%neural_tensor_layer_12/Sum_2:output:0%neural_tensor_layer_12/Sum_3:output:0%neural_tensor_layer_12/Sum_4:output:0%neural_tensor_layer_12/Sum_5:output:0%neural_tensor_layer_12/Sum_6:output:0%neural_tensor_layer_12/Sum_7:output:0%neural_tensor_layer_12/Sum_8:output:0%neural_tensor_layer_12/Sum_9:output:0&neural_tensor_layer_12/Sum_10:output:0&neural_tensor_layer_12/Sum_11:output:0&neural_tensor_layer_12/Sum_12:output:0&neural_tensor_layer_12/Sum_13:output:0&neural_tensor_layer_12/Sum_14:output:0&neural_tensor_layer_12/Sum_15:output:0-neural_tensor_layer_12/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2!
neural_tensor_layer_12/concat_1?
&neural_tensor_layer_12/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2(
&neural_tensor_layer_12/Reshape/shape/1?
$neural_tensor_layer_12/Reshape/shapePack-neural_tensor_layer_12/strided_slice:output:0/neural_tensor_layer_12/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2&
$neural_tensor_layer_12/Reshape/shape?
neural_tensor_layer_12/ReshapeReshape(neural_tensor_layer_12/concat_1:output:0-neural_tensor_layer_12/Reshape/shape:output:0*
T0*
_output_shapes

:2 
neural_tensor_layer_12/Reshape?
neural_tensor_layer_12/add_16AddV2'neural_tensor_layer_12/Reshape:output:0'neural_tensor_layer_12/MatMul:product:0*
T0*
_output_shapes

:2
neural_tensor_layer_12/add_16?
neural_tensor_layer_12/TanhTanh!neural_tensor_layer_12/add_16:z:0*
T0*
_output_shapes

:2
neural_tensor_layer_12/Tanh?
dense_48/MatMul/ReadVariableOpReadVariableOp'dense_48_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_48/MatMul/ReadVariableOp?
dense_48/MatMulMatMulneural_tensor_layer_12/Tanh:y:0&dense_48/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense_48/MatMul?
dense_48/BiasAdd/ReadVariableOpReadVariableOp(dense_48_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_48/BiasAdd/ReadVariableOp?
dense_48/BiasAddBiasAdddense_48/MatMul:product:0'dense_48/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense_48/BiasAddj
dense_48/ReluReludense_48/BiasAdd:output:0*
T0*
_output_shapes

:2
dense_48/Relu?
dense_49/MatMul/ReadVariableOpReadVariableOp'dense_49_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_49/MatMul/ReadVariableOp?
dense_49/MatMulMatMuldense_48/Relu:activations:0&dense_49/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense_49/MatMul?
dense_49/BiasAdd/ReadVariableOpReadVariableOp(dense_49_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_49/BiasAdd/ReadVariableOp?
dense_49/BiasAddBiasAdddense_49/MatMul:product:0'dense_49/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense_49/BiasAddj
dense_49/ReluReludense_49/BiasAdd:output:0*
T0*
_output_shapes

:2
dense_49/Relu?
dense_50/MatMul/ReadVariableOpReadVariableOp'dense_50_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_50/MatMul/ReadVariableOp?
dense_50/MatMulMatMuldense_49/Relu:activations:0&dense_50/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense_50/MatMul?
dense_50/BiasAdd/ReadVariableOpReadVariableOp(dense_50_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_50/BiasAdd/ReadVariableOp?
dense_50/BiasAddBiasAdddense_50/MatMul:product:0'dense_50/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense_50/BiasAddj
dense_50/ReluReludense_50/BiasAdd:output:0*
T0*
_output_shapes

:2
dense_50/Relu?
dense_51/MatMul/ReadVariableOpReadVariableOp'dense_51_matmul_readvariableop_resource*
_output_shapes

:*
dtype02 
dense_51/MatMul/ReadVariableOp?
dense_51/MatMulMatMuldense_50/Relu:activations:0&dense_51/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense_51/MatMul?
dense_51/BiasAdd/ReadVariableOpReadVariableOp(dense_51_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_51/BiasAdd/ReadVariableOp?
dense_51/BiasAddBiasAdddense_51/MatMul:product:0'dense_51/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense_51/BiasAdd?
tf.math.sigmoid_12/SigmoidSigmoiddense_51/BiasAdd:output:0*
T0*
_output_shapes

:2
tf.math.sigmoid_12/Sigmoidp
IdentityIdentitytf.math.sigmoid_12/Sigmoid:y:0^NoOp*
T0*
_output_shapes

:2

Identity?
NoOpNoOp#^attention_12/MatMul/ReadVariableOp%^attention_12/MatMul_3/ReadVariableOp ^dense_48/BiasAdd/ReadVariableOp^dense_48/MatMul/ReadVariableOp ^dense_49/BiasAdd/ReadVariableOp^dense_49/MatMul/ReadVariableOp ^dense_50/BiasAdd/ReadVariableOp^dense_50/MatMul/ReadVariableOp ^dense_51/BiasAdd/ReadVariableOp^dense_51/MatMul/ReadVariableOp#^graph_conv_36/add_1/ReadVariableOp#^graph_conv_36/add_5/ReadVariableOp'^graph_conv_36/transpose/ReadVariableOp)^graph_conv_36/transpose_2/ReadVariableOp#^graph_conv_37/add_1/ReadVariableOp#^graph_conv_37/add_5/ReadVariableOp'^graph_conv_37/transpose/ReadVariableOp)^graph_conv_37/transpose_2/ReadVariableOp#^graph_conv_38/add_1/ReadVariableOp#^graph_conv_38/add_5/ReadVariableOp'^graph_conv_38/transpose/ReadVariableOp)^graph_conv_38/transpose_2/ReadVariableOp-^neural_tensor_layer_12/MatMul/ReadVariableOp&^neural_tensor_layer_12/ReadVariableOp(^neural_tensor_layer_12/ReadVariableOp_1)^neural_tensor_layer_12/ReadVariableOp_10)^neural_tensor_layer_12/ReadVariableOp_11)^neural_tensor_layer_12/ReadVariableOp_12)^neural_tensor_layer_12/ReadVariableOp_13)^neural_tensor_layer_12/ReadVariableOp_14)^neural_tensor_layer_12/ReadVariableOp_15(^neural_tensor_layer_12/ReadVariableOp_2(^neural_tensor_layer_12/ReadVariableOp_3(^neural_tensor_layer_12/ReadVariableOp_4(^neural_tensor_layer_12/ReadVariableOp_5(^neural_tensor_layer_12/ReadVariableOp_6(^neural_tensor_layer_12/ReadVariableOp_7(^neural_tensor_layer_12/ReadVariableOp_8(^neural_tensor_layer_12/ReadVariableOp_9*^neural_tensor_layer_12/add/ReadVariableOp,^neural_tensor_layer_12/add_1/ReadVariableOp-^neural_tensor_layer_12/add_10/ReadVariableOp-^neural_tensor_layer_12/add_11/ReadVariableOp-^neural_tensor_layer_12/add_12/ReadVariableOp-^neural_tensor_layer_12/add_13/ReadVariableOp-^neural_tensor_layer_12/add_14/ReadVariableOp-^neural_tensor_layer_12/add_15/ReadVariableOp,^neural_tensor_layer_12/add_2/ReadVariableOp,^neural_tensor_layer_12/add_3/ReadVariableOp,^neural_tensor_layer_12/add_4/ReadVariableOp,^neural_tensor_layer_12/add_5/ReadVariableOp,^neural_tensor_layer_12/add_6/ReadVariableOp,^neural_tensor_layer_12/add_7/ReadVariableOp,^neural_tensor_layer_12/add_8/ReadVariableOp,^neural_tensor_layer_12/add_9/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????	:'???????????????????????????:?????????	:'???????????????????????????: : : : : : : : : : : : : : : : : : 2H
"attention_12/MatMul/ReadVariableOp"attention_12/MatMul/ReadVariableOp2L
$attention_12/MatMul_3/ReadVariableOp$attention_12/MatMul_3/ReadVariableOp2B
dense_48/BiasAdd/ReadVariableOpdense_48/BiasAdd/ReadVariableOp2@
dense_48/MatMul/ReadVariableOpdense_48/MatMul/ReadVariableOp2B
dense_49/BiasAdd/ReadVariableOpdense_49/BiasAdd/ReadVariableOp2@
dense_49/MatMul/ReadVariableOpdense_49/MatMul/ReadVariableOp2B
dense_50/BiasAdd/ReadVariableOpdense_50/BiasAdd/ReadVariableOp2@
dense_50/MatMul/ReadVariableOpdense_50/MatMul/ReadVariableOp2B
dense_51/BiasAdd/ReadVariableOpdense_51/BiasAdd/ReadVariableOp2@
dense_51/MatMul/ReadVariableOpdense_51/MatMul/ReadVariableOp2H
"graph_conv_36/add_1/ReadVariableOp"graph_conv_36/add_1/ReadVariableOp2H
"graph_conv_36/add_5/ReadVariableOp"graph_conv_36/add_5/ReadVariableOp2P
&graph_conv_36/transpose/ReadVariableOp&graph_conv_36/transpose/ReadVariableOp2T
(graph_conv_36/transpose_2/ReadVariableOp(graph_conv_36/transpose_2/ReadVariableOp2H
"graph_conv_37/add_1/ReadVariableOp"graph_conv_37/add_1/ReadVariableOp2H
"graph_conv_37/add_5/ReadVariableOp"graph_conv_37/add_5/ReadVariableOp2P
&graph_conv_37/transpose/ReadVariableOp&graph_conv_37/transpose/ReadVariableOp2T
(graph_conv_37/transpose_2/ReadVariableOp(graph_conv_37/transpose_2/ReadVariableOp2H
"graph_conv_38/add_1/ReadVariableOp"graph_conv_38/add_1/ReadVariableOp2H
"graph_conv_38/add_5/ReadVariableOp"graph_conv_38/add_5/ReadVariableOp2P
&graph_conv_38/transpose/ReadVariableOp&graph_conv_38/transpose/ReadVariableOp2T
(graph_conv_38/transpose_2/ReadVariableOp(graph_conv_38/transpose_2/ReadVariableOp2\
,neural_tensor_layer_12/MatMul/ReadVariableOp,neural_tensor_layer_12/MatMul/ReadVariableOp2N
%neural_tensor_layer_12/ReadVariableOp%neural_tensor_layer_12/ReadVariableOp2R
'neural_tensor_layer_12/ReadVariableOp_1'neural_tensor_layer_12/ReadVariableOp_12T
(neural_tensor_layer_12/ReadVariableOp_10(neural_tensor_layer_12/ReadVariableOp_102T
(neural_tensor_layer_12/ReadVariableOp_11(neural_tensor_layer_12/ReadVariableOp_112T
(neural_tensor_layer_12/ReadVariableOp_12(neural_tensor_layer_12/ReadVariableOp_122T
(neural_tensor_layer_12/ReadVariableOp_13(neural_tensor_layer_12/ReadVariableOp_132T
(neural_tensor_layer_12/ReadVariableOp_14(neural_tensor_layer_12/ReadVariableOp_142T
(neural_tensor_layer_12/ReadVariableOp_15(neural_tensor_layer_12/ReadVariableOp_152R
'neural_tensor_layer_12/ReadVariableOp_2'neural_tensor_layer_12/ReadVariableOp_22R
'neural_tensor_layer_12/ReadVariableOp_3'neural_tensor_layer_12/ReadVariableOp_32R
'neural_tensor_layer_12/ReadVariableOp_4'neural_tensor_layer_12/ReadVariableOp_42R
'neural_tensor_layer_12/ReadVariableOp_5'neural_tensor_layer_12/ReadVariableOp_52R
'neural_tensor_layer_12/ReadVariableOp_6'neural_tensor_layer_12/ReadVariableOp_62R
'neural_tensor_layer_12/ReadVariableOp_7'neural_tensor_layer_12/ReadVariableOp_72R
'neural_tensor_layer_12/ReadVariableOp_8'neural_tensor_layer_12/ReadVariableOp_82R
'neural_tensor_layer_12/ReadVariableOp_9'neural_tensor_layer_12/ReadVariableOp_92V
)neural_tensor_layer_12/add/ReadVariableOp)neural_tensor_layer_12/add/ReadVariableOp2Z
+neural_tensor_layer_12/add_1/ReadVariableOp+neural_tensor_layer_12/add_1/ReadVariableOp2\
,neural_tensor_layer_12/add_10/ReadVariableOp,neural_tensor_layer_12/add_10/ReadVariableOp2\
,neural_tensor_layer_12/add_11/ReadVariableOp,neural_tensor_layer_12/add_11/ReadVariableOp2\
,neural_tensor_layer_12/add_12/ReadVariableOp,neural_tensor_layer_12/add_12/ReadVariableOp2\
,neural_tensor_layer_12/add_13/ReadVariableOp,neural_tensor_layer_12/add_13/ReadVariableOp2\
,neural_tensor_layer_12/add_14/ReadVariableOp,neural_tensor_layer_12/add_14/ReadVariableOp2\
,neural_tensor_layer_12/add_15/ReadVariableOp,neural_tensor_layer_12/add_15/ReadVariableOp2Z
+neural_tensor_layer_12/add_2/ReadVariableOp+neural_tensor_layer_12/add_2/ReadVariableOp2Z
+neural_tensor_layer_12/add_3/ReadVariableOp+neural_tensor_layer_12/add_3/ReadVariableOp2Z
+neural_tensor_layer_12/add_4/ReadVariableOp+neural_tensor_layer_12/add_4/ReadVariableOp2Z
+neural_tensor_layer_12/add_5/ReadVariableOp+neural_tensor_layer_12/add_5/ReadVariableOp2Z
+neural_tensor_layer_12/add_6/ReadVariableOp+neural_tensor_layer_12/add_6/ReadVariableOp2Z
+neural_tensor_layer_12/add_7/ReadVariableOp+neural_tensor_layer_12/add_7/ReadVariableOp2Z
+neural_tensor_layer_12/add_8/ReadVariableOp+neural_tensor_layer_12/add_8/ReadVariableOp2Z
+neural_tensor_layer_12/add_9/ReadVariableOp+neural_tensor_layer_12/add_9/ReadVariableOp:U Q
+
_output_shapes
:?????????	
"
_user_specified_name
inputs/0:gc
=
_output_shapes+
):'???????????????????????????
"
_user_specified_name
inputs/1:UQ
+
_output_shapes
:?????????	
"
_user_specified_name
inputs/2:gc
=
_output_shapes+
):'???????????????????????????
"
_user_specified_name
inputs/3
?,
?
J__inference_graph_conv_37_layer_call_and_return_conditional_losses_4265128
inputs_0
inputs_11
shape_2_readvariableop_resource:@ +
add_1_readvariableop_resource: 
identity??add_1/ReadVariableOp?transpose/ReadVariableOp}
MatMulBatchMatMulV2inputs_1inputs_1*
T0*=
_output_shapes+
):'???????????????????????????2
MatMulM
ShapeShapeMatMul:output:0*
T0*
_output_shapes
:2
Shapev
addAddV2MatMul:output:0inputs_1*
T0*=
_output_shapes+
):'???????????????????????????2
add[
	Greater/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
	Greater/y?
GreaterGreateradd:z:0Greater/y:output:0*
T0*=
_output_shapes+
):'???????????????????????????2	
Greaterx
CastCastGreater:z:0*

DstT0*

SrcT0
*=
_output_shapes+
):'???????????????????????????2
CastJ
Shape_1Shapeinputs_0*
T0*
_output_shapes
:2	
Shape_1^
unstackUnpackShape_1:output:0*
T0*
_output_shapes
: : : *	
num2	
unstack?
Shape_2/ReadVariableOpReadVariableOpshape_2_readvariableop_resource*
_output_shapes

:@ *
dtype02
Shape_2/ReadVariableOpc
Shape_2Const*
_output_shapes
:*
dtype0*
valueB"@       2	
Shape_2`
	unstack_1UnpackShape_2:output:0*
T0*
_output_shapes
: : *	
num2
	unstack_1o
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"????@   2
Reshape/shapeq
ReshapeReshapeinputs_0Reshape/shape:output:0*
T0*'
_output_shapes
:?????????@2	
Reshape?
transpose/ReadVariableOpReadVariableOpshape_2_readvariableop_resource*
_output_shapes

:@ *
dtype02
transpose/ReadVariableOpq
transpose/permConst*
_output_shapes
:*
dtype0*
valueB"       2
transpose/perm?
	transpose	Transpose transpose/ReadVariableOp:value:0transpose/perm:output:0*
T0*
_output_shapes

:@ 2
	transposes
Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"@   ????2
Reshape_1/shapes
	Reshape_1Reshapetranspose:y:0Reshape_1/shape:output:0*
T0*
_output_shapes

:@ 2
	Reshape_1v
MatMul_1MatMulReshape:output:0Reshape_1:output:0*
T0*'
_output_shapes
:????????? 2

MatMul_1h
Reshape_2/shape/2Const*
_output_shapes
: *
dtype0*
value	B : 2
Reshape_2/shape/2?
Reshape_2/shapePackunstack:output:0unstack:output:1Reshape_2/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape_2/shape?
	Reshape_2ReshapeMatMul_1:product:0Reshape_2/shape:output:0*
T0*4
_output_shapes"
 :?????????????????? 2
	Reshape_2?
add_1/ReadVariableOpReadVariableOpadd_1_readvariableop_resource*
_output_shapes
: *
dtype02
add_1/ReadVariableOp?
add_1AddV2Reshape_2:output:0add_1/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :?????????????????? 2
add_1?
MatMul_2BatchMatMulV2Cast:y:0Cast:y:0*
T0*=
_output_shapes+
):'???????????????????????????2

MatMul_2S
Shape_3ShapeMatMul_2:output:0*
T0*
_output_shapes
:2	
Shape_3|
add_2AddV2MatMul_2:output:0Cast:y:0*
T0*=
_output_shapes+
):'???????????????????????????2
add_2_
Greater_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
Greater_1/y?
	Greater_1Greater	add_2:z:0Greater_1/y:output:0*
T0*=
_output_shapes+
):'???????????????????????????2
	Greater_1~
Cast_1CastGreater_1:z:0*

DstT0*

SrcT0
*=
_output_shapes+
):'???????????????????????????2
Cast_1y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose
Cast_1:y:0transpose_1/perm:output:0*
T0*=
_output_shapes+
):'???????????????????????????2
transpose_1?
MatMul_3BatchMatMulV2transpose_1:y:0	add_1:z:0*
T0*4
_output_shapes"
 :?????????????????? 2

MatMul_3S
Shape_4ShapeMatMul_3:output:0*
T0*
_output_shapes
:2	
Shape_4p
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indices?
SumSum
Cast_1:y:0Sum/reduction_indices:output:0*
T0*4
_output_shapes"
 :??????????????????*
	keep_dims(2
SumW
add_3/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32	
add_3/yv
add_3AddV2Sum:output:0add_3/y:output:0*
T0*4
_output_shapes"
 :??????????????????2
add_3z
truedivRealDivMatMul_3:output:0	add_3:z:0*
T0*4
_output_shapes"
 :?????????????????? 2	
truediv`
ReluRelutruediv:z:0*
T0*4
_output_shapes"
 :?????????????????? 2
Reluz
IdentityIdentityRelu:activations:0^NoOp*
T0*4
_output_shapes"
 :?????????????????? 2

Identity?
NoOpNoOp^add_1/ReadVariableOp^transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*`
_input_shapesO
M:??????????????????@:'???????????????????????????: : 2,
add_1/ReadVariableOpadd_1/ReadVariableOp24
transpose/ReadVariableOptranspose/ReadVariableOp:^ Z
4
_output_shapes"
 :??????????????????@
"
_user_specified_name
inputs/0:gc
=
_output_shapes+
):'???????????????????????????
"
_user_specified_name
inputs/1
?

?
E__inference_dense_48_layer_call_and_return_conditional_losses_4262954

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOpj
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpx
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2	
BiasAddO
ReluReluBiasAdd:output:0*
T0*
_output_shapes

:2
Relud
IdentityIdentityRelu:activations:0^NoOp*
T0*
_output_shapes

:2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
:: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:F B

_output_shapes

:
 
_user_specified_nameinputs
?]
?

E__inference_model_12_layer_call_and_return_conditional_losses_4263602
input_49
input_50
input_51
input_52'
graph_conv_36_4263536:@#
graph_conv_36_4263538:@'
graph_conv_37_4263544:@ #
graph_conv_37_4263546: '
graph_conv_38_4263552: #
graph_conv_38_4263554:&
attention_12_4263568:0
neural_tensor_layer_12_4263573: 4
neural_tensor_layer_12_4263575:,
neural_tensor_layer_12_4263577:"
dense_48_4263580:
dense_48_4263582:"
dense_49_4263585:
dense_49_4263587:"
dense_50_4263590:
dense_50_4263592:"
dense_51_4263595:
dense_51_4263597:
identity??$attention_12/StatefulPartitionedCall?&attention_12/StatefulPartitionedCall_1? dense_48/StatefulPartitionedCall? dense_49/StatefulPartitionedCall? dense_50/StatefulPartitionedCall? dense_51/StatefulPartitionedCall?%graph_conv_36/StatefulPartitionedCall?'graph_conv_36/StatefulPartitionedCall_1?%graph_conv_37/StatefulPartitionedCall?'graph_conv_37/StatefulPartitionedCall_1?%graph_conv_38/StatefulPartitionedCall?'graph_conv_38/StatefulPartitionedCall_1?.neural_tensor_layer_12/StatefulPartitionedCall?
%graph_conv_36/StatefulPartitionedCallStatefulPartitionedCallinput_51input_52graph_conv_36_4263536graph_conv_36_4263538*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_graph_conv_36_layer_call_and_return_conditional_losses_42625732'
%graph_conv_36/StatefulPartitionedCall?
'graph_conv_36/StatefulPartitionedCall_1StatefulPartitionedCallinput_49input_50graph_conv_36_4263536graph_conv_36_4263538*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_graph_conv_36_layer_call_and_return_conditional_losses_42625732)
'graph_conv_36/StatefulPartitionedCall_1?
%graph_conv_37/StatefulPartitionedCallStatefulPartitionedCall.graph_conv_36/StatefulPartitionedCall:output:0input_52graph_conv_37_4263544graph_conv_37_4263546*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :?????????????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_graph_conv_37_layer_call_and_return_conditional_losses_42626322'
%graph_conv_37/StatefulPartitionedCall?
'graph_conv_37/StatefulPartitionedCall_1StatefulPartitionedCall0graph_conv_36/StatefulPartitionedCall_1:output:0input_50graph_conv_37_4263544graph_conv_37_4263546*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :?????????????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_graph_conv_37_layer_call_and_return_conditional_losses_42626322)
'graph_conv_37/StatefulPartitionedCall_1?
%graph_conv_38/StatefulPartitionedCallStatefulPartitionedCall.graph_conv_37/StatefulPartitionedCall:output:0input_52graph_conv_38_4263552graph_conv_38_4263554*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_graph_conv_38_layer_call_and_return_conditional_losses_42626912'
%graph_conv_38/StatefulPartitionedCall?
'graph_conv_38/StatefulPartitionedCall_1StatefulPartitionedCall0graph_conv_37/StatefulPartitionedCall_1:output:0input_50graph_conv_38_4263552graph_conv_38_4263554*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_graph_conv_38_layer_call_and_return_conditional_losses_42626912)
'graph_conv_38/StatefulPartitionedCall_1?
/tf.__operators__.getitem_25/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/tf.__operators__.getitem_25/strided_slice/stack?
1tf.__operators__.getitem_25/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1tf.__operators__.getitem_25/strided_slice/stack_1?
1tf.__operators__.getitem_25/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1tf.__operators__.getitem_25/strided_slice/stack_2?
)tf.__operators__.getitem_25/strided_sliceStridedSlice.graph_conv_38/StatefulPartitionedCall:output:08tf.__operators__.getitem_25/strided_slice/stack:output:0:tf.__operators__.getitem_25/strided_slice/stack_1:output:0:tf.__operators__.getitem_25/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2+
)tf.__operators__.getitem_25/strided_slice?
/tf.__operators__.getitem_24/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/tf.__operators__.getitem_24/strided_slice/stack?
1tf.__operators__.getitem_24/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1tf.__operators__.getitem_24/strided_slice/stack_1?
1tf.__operators__.getitem_24/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1tf.__operators__.getitem_24/strided_slice/stack_2?
)tf.__operators__.getitem_24/strided_sliceStridedSlice0graph_conv_38/StatefulPartitionedCall_1:output:08tf.__operators__.getitem_24/strided_slice/stack:output:0:tf.__operators__.getitem_24/strided_slice/stack_1:output:0:tf.__operators__.getitem_24/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
shrink_axis_mask2+
)tf.__operators__.getitem_24/strided_slice?
$attention_12/StatefulPartitionedCallStatefulPartitionedCall2tf.__operators__.getitem_24/strided_slice:output:0attention_12_4263568*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_attention_12_layer_call_and_return_conditional_losses_42627292&
$attention_12/StatefulPartitionedCall?
&attention_12/StatefulPartitionedCall_1StatefulPartitionedCall2tf.__operators__.getitem_25/strided_slice:output:0attention_12_4263568*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_attention_12_layer_call_and_return_conditional_losses_42627292(
&attention_12/StatefulPartitionedCall_1?
.neural_tensor_layer_12/StatefulPartitionedCallStatefulPartitionedCall-attention_12/StatefulPartitionedCall:output:0/attention_12/StatefulPartitionedCall_1:output:0neural_tensor_layer_12_4263573neural_tensor_layer_12_4263575neural_tensor_layer_12_4263577*
Tin	
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *\
fWRU
S__inference_neural_tensor_layer_12_layer_call_and_return_conditional_losses_426293520
.neural_tensor_layer_12/StatefulPartitionedCall?
 dense_48/StatefulPartitionedCallStatefulPartitionedCall7neural_tensor_layer_12/StatefulPartitionedCall:output:0dense_48_4263580dense_48_4263582*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_48_layer_call_and_return_conditional_losses_42629542"
 dense_48/StatefulPartitionedCall?
 dense_49/StatefulPartitionedCallStatefulPartitionedCall)dense_48/StatefulPartitionedCall:output:0dense_49_4263585dense_49_4263587*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_49_layer_call_and_return_conditional_losses_42629712"
 dense_49/StatefulPartitionedCall?
 dense_50/StatefulPartitionedCallStatefulPartitionedCall)dense_49/StatefulPartitionedCall:output:0dense_50_4263590dense_50_4263592*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_50_layer_call_and_return_conditional_losses_42629882"
 dense_50/StatefulPartitionedCall?
 dense_51/StatefulPartitionedCallStatefulPartitionedCall)dense_50/StatefulPartitionedCall:output:0dense_51_4263595dense_51_4263597*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_dense_51_layer_call_and_return_conditional_losses_42630042"
 dense_51/StatefulPartitionedCall?
tf.math.sigmoid_12/SigmoidSigmoid)dense_51/StatefulPartitionedCall:output:0*
T0*
_output_shapes

:2
tf.math.sigmoid_12/Sigmoidp
IdentityIdentitytf.math.sigmoid_12/Sigmoid:y:0^NoOp*
T0*
_output_shapes

:2

Identity?
NoOpNoOp%^attention_12/StatefulPartitionedCall'^attention_12/StatefulPartitionedCall_1!^dense_48/StatefulPartitionedCall!^dense_49/StatefulPartitionedCall!^dense_50/StatefulPartitionedCall!^dense_51/StatefulPartitionedCall&^graph_conv_36/StatefulPartitionedCall(^graph_conv_36/StatefulPartitionedCall_1&^graph_conv_37/StatefulPartitionedCall(^graph_conv_37/StatefulPartitionedCall_1&^graph_conv_38/StatefulPartitionedCall(^graph_conv_38/StatefulPartitionedCall_1/^neural_tensor_layer_12/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????	:'???????????????????????????:?????????	:'???????????????????????????: : : : : : : : : : : : : : : : : : 2L
$attention_12/StatefulPartitionedCall$attention_12/StatefulPartitionedCall2P
&attention_12/StatefulPartitionedCall_1&attention_12/StatefulPartitionedCall_12D
 dense_48/StatefulPartitionedCall dense_48/StatefulPartitionedCall2D
 dense_49/StatefulPartitionedCall dense_49/StatefulPartitionedCall2D
 dense_50/StatefulPartitionedCall dense_50/StatefulPartitionedCall2D
 dense_51/StatefulPartitionedCall dense_51/StatefulPartitionedCall2N
%graph_conv_36/StatefulPartitionedCall%graph_conv_36/StatefulPartitionedCall2R
'graph_conv_36/StatefulPartitionedCall_1'graph_conv_36/StatefulPartitionedCall_12N
%graph_conv_37/StatefulPartitionedCall%graph_conv_37/StatefulPartitionedCall2R
'graph_conv_37/StatefulPartitionedCall_1'graph_conv_37/StatefulPartitionedCall_12N
%graph_conv_38/StatefulPartitionedCall%graph_conv_38/StatefulPartitionedCall2R
'graph_conv_38/StatefulPartitionedCall_1'graph_conv_38/StatefulPartitionedCall_12`
.neural_tensor_layer_12/StatefulPartitionedCall.neural_tensor_layer_12/StatefulPartitionedCall:U Q
+
_output_shapes
:?????????	
"
_user_specified_name
input_49:gc
=
_output_shapes+
):'???????????????????????????
"
_user_specified_name
input_50:UQ
+
_output_shapes
:?????????	
"
_user_specified_name
input_51:gc
=
_output_shapes+
):'???????????????????????????
"
_user_specified_name
input_52
?	
?
/__inference_graph_conv_36_layer_call_fn_4264906
inputs_0
inputs_1
unknown:@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :??????????????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_graph_conv_36_layer_call_and_return_conditional_losses_42633192
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :??????????????????@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*W
_input_shapesF
D:?????????	:'???????????????????????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:?????????	
"
_user_specified_name
inputs/0:gc
=
_output_shapes+
):'???????????????????????????
"
_user_specified_name
inputs/1
?	
?
/__inference_graph_conv_37_layer_call_fn_4265018
inputs_0
inputs_1
unknown:@ 
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :?????????????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_graph_conv_37_layer_call_and_return_conditional_losses_42626322
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :?????????????????? 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*`
_input_shapesO
M:??????????????????@:'???????????????????????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :??????????????????@
"
_user_specified_name
inputs/0:gc
=
_output_shapes+
):'???????????????????????????
"
_user_specified_name
inputs/1"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
A
input_495
serving_default_input_49:0?????????	
S
input_50G
serving_default_input_50:0'???????????????????????????
A
input_515
serving_default_input_51:0?????????	
S
input_52G
serving_default_input_52:0'???????????????????????????=
tf.math.sigmoid_12'
StatefulPartitionedCall:0tensorflow/serving/predict:??
?
layer-0
layer-1
layer-2
layer-3
layer_with_weights-0
layer-4
layer_with_weights-1
layer-5
layer_with_weights-2
layer-6
layer-7
	layer-8

layer_with_weights-3

layer-9
layer_with_weights-4
layer-10
layer_with_weights-5
layer-11
layer_with_weights-6
layer-12
layer_with_weights-7
layer-13
layer_with_weights-8
layer-14
layer-15
	optimizer
trainable_variables
	variables
regularization_losses
	keras_api

signatures
?__call__
+?&call_and_return_all_conditional_losses
?_default_save_signature"
_tf_keras_network
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
?
graph_conv_36_W
W
graph_conv_36_b
b
trainable_variables
	variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
graph_conv_37_W
W
graph_conv_37_b
b
trainable_variables
 	variables
!regularization_losses
"	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
#graph_conv_38_W
#W
$graph_conv_38_b
$b
%trainable_variables
&	variables
'regularization_losses
(	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
(
)	keras_api"
_tf_keras_layer
(
*	keras_api"
_tf_keras_layer
?
+
att_weight
+weights_att
,trainable_variables
-	variables
.regularization_losses
/	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
0W
1V
2b
3trainable_weights2
4trainable_variables
5	variables
6regularization_losses
7	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

8kernel
9bias
:trainable_variables
;	variables
<regularization_losses
=	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

>kernel
?bias
@trainable_variables
A	variables
Bregularization_losses
C	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

Dkernel
Ebias
Ftrainable_variables
G	variables
Hregularization_losses
I	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

Jkernel
Kbias
Ltrainable_variables
M	variables
Nregularization_losses
O	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
(
P	keras_api"
_tf_keras_layer
?
Qiter
	Rdecay
Slearning_rate
Trho
accum_grad?
accum_grad?
accum_grad?
accum_grad?#
accum_grad?$
accum_grad?+
accum_grad?0
accum_grad?1
accum_grad?2
accum_grad?8
accum_grad?9
accum_grad?>
accum_grad??
accum_grad?D
accum_grad?E
accum_grad?J
accum_grad?K
accum_grad?	accum_var?	accum_var?	accum_var?	accum_var?#	accum_var?$	accum_var?+	accum_var?0	accum_var?1	accum_var?2	accum_var?8	accum_var?9	accum_var?>	accum_var??	accum_var?D	accum_var?E	accum_var?J	accum_var?K	accum_var?"
	optimizer
?
0
1
2
3
#4
$5
+6
07
18
29
810
911
>12
?13
D14
E15
J16
K17"
trackable_list_wrapper
?
0
1
2
3
#4
$5
+6
07
18
29
810
911
>12
?13
D14
E15
J16
K17"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Unon_trainable_variables
trainable_variables
	variables
regularization_losses
Vlayer_metrics
Wlayer_regularization_losses

Xlayers
Ymetrics
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
/:-@2graph_conv_36/graph_conv_36_W
+:)@2graph_conv_36/graph_conv_36_b
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Znon_trainable_variables
trainable_variables
	variables
regularization_losses
[layer_metrics
\layer_regularization_losses

]layers
^metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
/:-@ 2graph_conv_37/graph_conv_37_W
+:) 2graph_conv_37/graph_conv_37_b
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
_non_trainable_variables
trainable_variables
 	variables
!regularization_losses
`layer_metrics
alayer_regularization_losses

blayers
cmetrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
/:- 2graph_conv_38/graph_conv_38_W
+:)2graph_conv_38/graph_conv_38_b
.
#0
$1"
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
dnon_trainable_variables
%trainable_variables
&	variables
'regularization_losses
elayer_metrics
flayer_regularization_losses

glayers
hmetrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
"
_generic_user_object
):'2attention_12/att_weight
'
+0"
trackable_list_wrapper
'
+0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
inon_trainable_variables
,trainable_variables
-	variables
.regularization_losses
jlayer_metrics
klayer_regularization_losses

llayers
mmetrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
5:32neural_tensor_layer_12/Variable
1:/ 2neural_tensor_layer_12/Variable
-:+2neural_tensor_layer_12/Variable
5
00
11
22"
trackable_list_wrapper
5
00
11
22"
trackable_list_wrapper
5
00
11
22"
trackable_list_wrapper
 "
trackable_list_wrapper
?
nnon_trainable_variables
4trainable_variables
5	variables
6regularization_losses
olayer_metrics
player_regularization_losses

qlayers
rmetrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:2dense_48/kernel
:2dense_48/bias
.
80
91"
trackable_list_wrapper
.
80
91"
trackable_list_wrapper
 "
trackable_list_wrapper
?
snon_trainable_variables
:trainable_variables
;	variables
<regularization_losses
tlayer_metrics
ulayer_regularization_losses

vlayers
wmetrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:2dense_49/kernel
:2dense_49/bias
.
>0
?1"
trackable_list_wrapper
.
>0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
xnon_trainable_variables
@trainable_variables
A	variables
Bregularization_losses
ylayer_metrics
zlayer_regularization_losses

{layers
|metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:2dense_50/kernel
:2dense_50/bias
.
D0
E1"
trackable_list_wrapper
.
D0
E1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
}non_trainable_variables
Ftrainable_variables
G	variables
Hregularization_losses
~layer_metrics
layer_regularization_losses
?layers
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:2dense_51/kernel
:2dense_51/bias
.
J0
K1"
trackable_list_wrapper
.
J0
K1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
Ltrainable_variables
M	variables
Nregularization_losses
?layer_metrics
 ?layer_regularization_losses
?layers
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
:	 (2Adadelta/iter
: (2Adadelta/decay
 : (2Adadelta/learning_rate
: (2Adadelta/rho
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
R

?total

?count
?	variables
?	keras_api"
_tf_keras_metric
c

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"
_tf_keras_metric
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
A:?@21Adadelta/graph_conv_36/graph_conv_36_W/accum_grad
=:;@21Adadelta/graph_conv_36/graph_conv_36_b/accum_grad
A:?@ 21Adadelta/graph_conv_37/graph_conv_37_W/accum_grad
=:; 21Adadelta/graph_conv_37/graph_conv_37_b/accum_grad
A:? 21Adadelta/graph_conv_38/graph_conv_38_W/accum_grad
=:;21Adadelta/graph_conv_38/graph_conv_38_b/accum_grad
;:92+Adadelta/attention_12/att_weight/accum_grad
G:E23Adadelta/neural_tensor_layer_12/Variable/accum_grad
C:A 23Adadelta/neural_tensor_layer_12/Variable/accum_grad
?:=23Adadelta/neural_tensor_layer_12/Variable/accum_grad
3:12#Adadelta/dense_48/kernel/accum_grad
-:+2!Adadelta/dense_48/bias/accum_grad
3:12#Adadelta/dense_49/kernel/accum_grad
-:+2!Adadelta/dense_49/bias/accum_grad
3:12#Adadelta/dense_50/kernel/accum_grad
-:+2!Adadelta/dense_50/bias/accum_grad
3:12#Adadelta/dense_51/kernel/accum_grad
-:+2!Adadelta/dense_51/bias/accum_grad
@:>@20Adadelta/graph_conv_36/graph_conv_36_W/accum_var
<::@20Adadelta/graph_conv_36/graph_conv_36_b/accum_var
@:>@ 20Adadelta/graph_conv_37/graph_conv_37_W/accum_var
<:: 20Adadelta/graph_conv_37/graph_conv_37_b/accum_var
@:> 20Adadelta/graph_conv_38/graph_conv_38_W/accum_var
<::20Adadelta/graph_conv_38/graph_conv_38_b/accum_var
::82*Adadelta/attention_12/att_weight/accum_var
F:D22Adadelta/neural_tensor_layer_12/Variable/accum_var
B:@ 22Adadelta/neural_tensor_layer_12/Variable/accum_var
>:<22Adadelta/neural_tensor_layer_12/Variable/accum_var
2:02"Adadelta/dense_48/kernel/accum_var
,:*2 Adadelta/dense_48/bias/accum_var
2:02"Adadelta/dense_49/kernel/accum_var
,:*2 Adadelta/dense_49/bias/accum_var
2:02"Adadelta/dense_50/kernel/accum_var
,:*2 Adadelta/dense_50/bias/accum_var
2:02"Adadelta/dense_51/kernel/accum_var
,:*2 Adadelta/dense_51/bias/accum_var
?2?
*__inference_model_12_layer_call_fn_4263051
*__inference_model_12_layer_call_fn_4263768
*__inference_model_12_layer_call_fn_4263812
*__inference_model_12_layer_call_fn_4263530?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
E__inference_model_12_layer_call_and_return_conditional_losses_4264349
E__inference_model_12_layer_call_and_return_conditional_losses_4264886
E__inference_model_12_layer_call_and_return_conditional_losses_4263602
E__inference_model_12_layer_call_and_return_conditional_losses_4263674?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
"__inference__wrapped_model_4262509input_49input_50input_51input_52"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
/__inference_graph_conv_36_layer_call_fn_4264896
/__inference_graph_conv_36_layer_call_fn_4264906?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?2?
J__inference_graph_conv_36_layer_call_and_return_conditional_losses_4264957
J__inference_graph_conv_36_layer_call_and_return_conditional_losses_4265008?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?2?
/__inference_graph_conv_37_layer_call_fn_4265018
/__inference_graph_conv_37_layer_call_fn_4265028?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?2?
J__inference_graph_conv_37_layer_call_and_return_conditional_losses_4265078
J__inference_graph_conv_37_layer_call_and_return_conditional_losses_4265128?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?2?
/__inference_graph_conv_38_layer_call_fn_4265138
/__inference_graph_conv_38_layer_call_fn_4265148?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?2?
J__inference_graph_conv_38_layer_call_and_return_conditional_losses_4265198
J__inference_graph_conv_38_layer_call_and_return_conditional_losses_4265248?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?2?
.__inference_attention_12_layer_call_fn_4265255?
???
FullArgSpec 
args?
jself
j	embedding
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
I__inference_attention_12_layer_call_and_return_conditional_losses_4265276?
???
FullArgSpec 
args?
jself
j	embedding
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
8__inference_neural_tensor_layer_12_layer_call_fn_4265288?
???
FullArgSpec%
args?
jself
jinputs
jmask
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
S__inference_neural_tensor_layer_12_layer_call_and_return_conditional_losses_4265488?
???
FullArgSpec%
args?
jself
jinputs
jmask
varargs
 
varkw
 
defaults?

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_dense_48_layer_call_fn_4265497?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_dense_48_layer_call_and_return_conditional_losses_4265508?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_dense_49_layer_call_fn_4265517?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_dense_49_layer_call_and_return_conditional_losses_4265528?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_dense_50_layer_call_fn_4265537?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_dense_50_layer_call_and_return_conditional_losses_4265548?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_dense_51_layer_call_fn_4265557?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_dense_51_layer_call_and_return_conditional_losses_4265567?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
%__inference_signature_wrapper_4263724input_49input_50input_51input_52"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
"__inference__wrapped_model_4262509?#$+10289>?DEJK???
???
???
&?#
input_49?????????	
8?5
input_50'???????????????????????????
&?#
input_51?????????	
8?5
input_52'???????????????????????????
? ">?;
9
tf.math.sigmoid_12#? 
tf.math.sigmoid_12?
I__inference_attention_12_layer_call_and_return_conditional_losses_4265276U+2?/
(?%
#? 
	embedding?????????
? "?
?
0
? z
.__inference_attention_12_layer_call_fn_4265255H+2?/
(?%
#? 
	embedding?????????
? "??
E__inference_dense_48_layer_call_and_return_conditional_losses_4265508J89&?#
?
?
inputs
? "?
?
0
? k
*__inference_dense_48_layer_call_fn_4265497=89&?#
?
?
inputs
? "??
E__inference_dense_49_layer_call_and_return_conditional_losses_4265528J>?&?#
?
?
inputs
? "?
?
0
? k
*__inference_dense_49_layer_call_fn_4265517=>?&?#
?
?
inputs
? "??
E__inference_dense_50_layer_call_and_return_conditional_losses_4265548JDE&?#
?
?
inputs
? "?
?
0
? k
*__inference_dense_50_layer_call_fn_4265537=DE&?#
?
?
inputs
? "??
E__inference_dense_51_layer_call_and_return_conditional_losses_4265567JJK&?#
?
?
inputs
? "?
?
0
? k
*__inference_dense_51_layer_call_fn_4265557=JK&?#
?
?
inputs
? "??
J__inference_graph_conv_36_layer_call_and_return_conditional_losses_4264957????
j?g
e?b
&?#
inputs/0?????????	
8?5
inputs/1'???????????????????????????
?

trainingp "2?/
(?%
0??????????????????@
? ?
J__inference_graph_conv_36_layer_call_and_return_conditional_losses_4265008????
j?g
e?b
&?#
inputs/0?????????	
8?5
inputs/1'???????????????????????????
?

trainingp"2?/
(?%
0??????????????????@
? ?
/__inference_graph_conv_36_layer_call_fn_4264896????
j?g
e?b
&?#
inputs/0?????????	
8?5
inputs/1'???????????????????????????
?

trainingp "%?"??????????????????@?
/__inference_graph_conv_36_layer_call_fn_4264906????
j?g
e?b
&?#
inputs/0?????????	
8?5
inputs/1'???????????????????????????
?

trainingp"%?"??????????????????@?
J__inference_graph_conv_37_layer_call_and_return_conditional_losses_4265078????
s?p
n?k
/?,
inputs/0??????????????????@
8?5
inputs/1'???????????????????????????
?

trainingp "2?/
(?%
0?????????????????? 
? ?
J__inference_graph_conv_37_layer_call_and_return_conditional_losses_4265128????
s?p
n?k
/?,
inputs/0??????????????????@
8?5
inputs/1'???????????????????????????
?

trainingp"2?/
(?%
0?????????????????? 
? ?
/__inference_graph_conv_37_layer_call_fn_4265018????
s?p
n?k
/?,
inputs/0??????????????????@
8?5
inputs/1'???????????????????????????
?

trainingp "%?"?????????????????? ?
/__inference_graph_conv_37_layer_call_fn_4265028????
s?p
n?k
/?,
inputs/0??????????????????@
8?5
inputs/1'???????????????????????????
?

trainingp"%?"?????????????????? ?
J__inference_graph_conv_38_layer_call_and_return_conditional_losses_4265198?#$???
s?p
n?k
/?,
inputs/0?????????????????? 
8?5
inputs/1'???????????????????????????
?

trainingp "2?/
(?%
0??????????????????
? ?
J__inference_graph_conv_38_layer_call_and_return_conditional_losses_4265248?#$???
s?p
n?k
/?,
inputs/0?????????????????? 
8?5
inputs/1'???????????????????????????
?

trainingp"2?/
(?%
0??????????????????
? ?
/__inference_graph_conv_38_layer_call_fn_4265138?#$???
s?p
n?k
/?,
inputs/0?????????????????? 
8?5
inputs/1'???????????????????????????
?

trainingp "%?"???????????????????
/__inference_graph_conv_38_layer_call_fn_4265148?#$???
s?p
n?k
/?,
inputs/0?????????????????? 
8?5
inputs/1'???????????????????????????
?

trainingp"%?"???????????????????
E__inference_model_12_layer_call_and_return_conditional_losses_4263602?#$+10289>?DEJK???
???
???
&?#
input_49?????????	
8?5
input_50'???????????????????????????
&?#
input_51?????????	
8?5
input_52'???????????????????????????
p 

 
? "?
?
0
? ?
E__inference_model_12_layer_call_and_return_conditional_losses_4263674?#$+10289>?DEJK???
???
???
&?#
input_49?????????	
8?5
input_50'???????????????????????????
&?#
input_51?????????	
8?5
input_52'???????????????????????????
p

 
? "?
?
0
? ?
E__inference_model_12_layer_call_and_return_conditional_losses_4264349?#$+10289>?DEJK???
???
???
&?#
inputs/0?????????	
8?5
inputs/1'???????????????????????????
&?#
inputs/2?????????	
8?5
inputs/3'???????????????????????????
p 

 
? "?
?
0
? ?
E__inference_model_12_layer_call_and_return_conditional_losses_4264886?#$+10289>?DEJK???
???
???
&?#
inputs/0?????????	
8?5
inputs/1'???????????????????????????
&?#
inputs/2?????????	
8?5
inputs/3'???????????????????????????
p

 
? "?
?
0
? ?
*__inference_model_12_layer_call_fn_4263051?#$+10289>?DEJK???
???
???
&?#
input_49?????????	
8?5
input_50'???????????????????????????
&?#
input_51?????????	
8?5
input_52'???????????????????????????
p 

 
? "??
*__inference_model_12_layer_call_fn_4263530?#$+10289>?DEJK???
???
???
&?#
input_49?????????	
8?5
input_50'???????????????????????????
&?#
input_51?????????	
8?5
input_52'???????????????????????????
p

 
? "??
*__inference_model_12_layer_call_fn_4263768?#$+10289>?DEJK???
???
???
&?#
inputs/0?????????	
8?5
inputs/1'???????????????????????????
&?#
inputs/2?????????	
8?5
inputs/3'???????????????????????????
p 

 
? "??
*__inference_model_12_layer_call_fn_4263812?#$+10289>?DEJK???
???
???
&?#
inputs/0?????????	
8?5
inputs/1'???????????????????????????
&?#
inputs/2?????????	
8?5
inputs/3'???????????????????????????
p

 
? "??
S__inference_neural_tensor_layer_12_layer_call_and_return_conditional_losses_4265488q102L?I
B??
9?6
?
inputs/0
?
inputs/1

 
? "?
?
0
? ?
8__inference_neural_tensor_layer_12_layer_call_fn_4265288d102L?I
B??
9?6
?
inputs/0
?
inputs/1

 
? "??
%__inference_signature_wrapper_4263724?#$+10289>?DEJK???
? 
???
2
input_49&?#
input_49?????????	
D
input_508?5
input_50'???????????????????????????
2
input_51&?#
input_51?????????	
D
input_528?5
input_52'???????????????????????????">?;
9
tf.math.sigmoid_12#? 
tf.math.sigmoid_12