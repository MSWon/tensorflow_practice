αε
¨**
:
Add
x"T
y"T
z"T"
Ttype:
2	
ξ
	ApplyAdam
var"T	
m"T	
v"T
beta1_power"T
beta2_power"T
lr"T

beta1"T

beta2"T
epsilon"T	
grad"T
out"T" 
Ttype:
2	"
use_lockingbool( "
use_nesterovbool( 

ArgMax

input"T
	dimension"Tidx
output"output_type" 
Ttype:
2	"
Tidxtype0:
2	"
output_typetype0	:
2	
x
Assign
ref"T

value"T

output_ref"T"	
Ttype"
validate_shapebool("
use_lockingbool(
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
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
Y
	DecodePng
contents
image"dtype"
channelsint "
dtypetype0:
2
y
Enter	
data"T
output"T"	
Ttype"

frame_namestring"
is_constantbool( "
parallel_iterationsint

B
Equal
x"T
y"T
z
"
Ttype:
2	

)
Exit	
data"T
output"T"	
Ttype
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
?
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
:
Less
x"T
y"T
z
"
Ttype:
2	
?

LogSoftmax
logits"T

logsoftmax"T"
Ttype:
2
$

LogicalAnd
x

y

z

!
LoopCond	
input


output

q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
8
Maximum
x"T
y"T
z"T"
Ttype:

2	

Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
N
Merge
inputs"T*N
output"T
value_index"	
Ttype"
Nint(0
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
=
Mul
x"T
y"T
z"T"
Ttype:
2	
.
Neg
x"T
y"T"
Ttype:

2	
2
NextIteration	
data"T
output"T"	
Ttype
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
X
PlaceholderWithDefault
input"dtype
output"dtype"
dtypetype"
shapeshape

Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	

RandomStandardNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
a
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:	
2	
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
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
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
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
a
Slice

input"T
begin"Index
size"Index
output"T"	
Ttype"
Indextype:
2	
9
Softmax
logits"T
softmax"T"
Ttype:
2
j
SoftmaxCrossEntropyWithLogits
features"T
labels"T	
loss"T
backprop"T"
Ttype:
2
φ
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
:
Sub
x"T
y"T
z"T"
Ttype:
2	

Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
M
Switch	
data"T
pred

output_false"T
output_true"T"	
Ttype
{
TensorArrayGatherV3

handle
indices
flow_in
value"dtype"
dtypetype"
element_shapeshape:
Y
TensorArrayReadV3

handle	
index
flow_in
value"dtype"
dtypetype
d
TensorArrayScatterV3

handle
indices

value"T
flow_in
flow_out"	
Ttype
9
TensorArraySizeV3

handle
flow_in
size
ή
TensorArrayV3
size

handle
flow"
dtypetype"
element_shapeshape:"
dynamic_sizebool( "
clear_after_readbool("$
identical_element_shapesbool( "
tensor_array_namestring 
`
TensorArrayWriteV3

handle	
index

value"T
flow_in
flow_out"	
Ttype
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
s

VariableV2
ref"dtype"
shapeshape"
dtypetype"
	containerstring "
shared_namestring 
&
	ZerosLike
x"T
y"T"	
Ttype"serve*1.14.02unknownχ¦
`
raw_xPlaceholder*
shape:?????????*
dtype0*#
_output_shapes
:?????????
N
	map/ShapeShaperaw_x*
_output_shapes
:*
T0*
out_type0
a
map/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
c
map/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
c
map/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:

map/strided_sliceStridedSlice	map/Shapemap/strided_slice/stackmap/strided_slice/stack_1map/strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
Ϊ
map/TensorArrayTensorArrayV3map/strided_slice*
element_shape:*
dynamic_size( *
clear_after_read(*
identical_element_shapes(*
tensor_array_name *
dtype0*
_output_shapes

:: 
a
map/TensorArrayUnstack/ShapeShaperaw_x*
T0*
out_type0*
_output_shapes
:
t
*map/TensorArrayUnstack/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
v
,map/TensorArrayUnstack/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
v
,map/TensorArrayUnstack/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
μ
$map/TensorArrayUnstack/strided_sliceStridedSlicemap/TensorArrayUnstack/Shape*map/TensorArrayUnstack/strided_slice/stack,map/TensorArrayUnstack/strided_slice/stack_1,map/TensorArrayUnstack/strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
: 
d
"map/TensorArrayUnstack/range/startConst*
dtype0*
_output_shapes
: *
value	B : 
d
"map/TensorArrayUnstack/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
Δ
map/TensorArrayUnstack/rangeRange"map/TensorArrayUnstack/range/start$map/TensorArrayUnstack/strided_slice"map/TensorArrayUnstack/range/delta*#
_output_shapes
:?????????*

Tidx0
Ϊ
>map/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3map/TensorArraymap/TensorArrayUnstack/rangeraw_xmap/TensorArray:1*
T0*
_class

loc:@raw_x*
_output_shapes
: 
K
	map/ConstConst*
value	B : *
dtype0*
_output_shapes
: 
ά
map/TensorArray_1TensorArrayV3map/strided_slice*
dtype0*
_output_shapes

:: *
element_shape:*
clear_after_read(*
dynamic_size( *
identical_element_shapes(*
tensor_array_name 
]
map/while/iteration_counterConst*
value	B : *
dtype0*
_output_shapes
: 
­
map/while/EnterEntermap/while/iteration_counter*
parallel_iterations
*
_output_shapes
: *'

frame_namemap/while/while_context*
T0*
is_constant( 

map/while/Enter_1Enter	map/Const*
parallel_iterations
*
_output_shapes
: *'

frame_namemap/while/while_context*
T0*
is_constant( 
§
map/while/Enter_2Entermap/TensorArray_1:1*
T0*
is_constant( *
parallel_iterations
*
_output_shapes
: *'

frame_namemap/while/while_context
n
map/while/MergeMergemap/while/Entermap/while/NextIteration*
N*
_output_shapes
: : *
T0
t
map/while/Merge_1Mergemap/while/Enter_1map/while/NextIteration_1*
T0*
N*
_output_shapes
: : 
t
map/while/Merge_2Mergemap/while/Enter_2map/while/NextIteration_2*
T0*
N*
_output_shapes
: : 
^
map/while/LessLessmap/while/Mergemap/while/Less/Enter*
T0*
_output_shapes
: 
¨
map/while/Less/EnterEntermap/strided_slice*
T0*
is_constant(*
parallel_iterations
*
_output_shapes
: *'

frame_namemap/while/while_context
b
map/while/Less_1Lessmap/while/Merge_1map/while/Less/Enter*
T0*
_output_shapes
: 
\
map/while/LogicalAnd
LogicalAndmap/while/Lessmap/while/Less_1*
_output_shapes
: 
L
map/while/LoopCondLoopCondmap/while/LogicalAnd*
_output_shapes
: 

map/while/SwitchSwitchmap/while/Mergemap/while/LoopCond*
T0*"
_class
loc:@map/while/Merge*
_output_shapes
: : 

map/while/Switch_1Switchmap/while/Merge_1map/while/LoopCond*
T0*$
_class
loc:@map/while/Merge_1*
_output_shapes
: : 

map/while/Switch_2Switchmap/while/Merge_2map/while/LoopCond*
_output_shapes
: : *
T0*$
_class
loc:@map/while/Merge_2
S
map/while/IdentityIdentitymap/while/Switch:1*
_output_shapes
: *
T0
W
map/while/Identity_1Identitymap/while/Switch_1:1*
T0*
_output_shapes
: 
W
map/while/Identity_2Identitymap/while/Switch_2:1*
T0*
_output_shapes
: 
f
map/while/add/yConst^map/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
Z
map/while/addAddmap/while/Identitymap/while/add/y*
T0*
_output_shapes
: 
³
map/while/TensorArrayReadV3TensorArrayReadV3!map/while/TensorArrayReadV3/Entermap/while/Identity_1#map/while/TensorArrayReadV3/Enter_1*
dtype0*
_output_shapes
: 
·
!map/while/TensorArrayReadV3/EnterEntermap/TensorArray*
T0*
is_constant(*
parallel_iterations
*
_output_shapes
:*'

frame_namemap/while/while_context
δ
#map/while/TensorArrayReadV3/Enter_1Enter>map/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
T0*
is_constant(*
parallel_iterations
*
_output_shapes
: *'

frame_namemap/while/while_context

map/while/DecodePng	DecodePngmap/while/TensorArrayReadV3*
channels*
dtype0*4
_output_shapes"
 :??????????????????

-map/while/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV33map/while/TensorArrayWrite/TensorArrayWriteV3/Entermap/while/Identity_1map/while/DecodePngmap/while/Identity_2*
T0*&
_class
loc:@map/while/DecodePng*
_output_shapes
: 
σ
3map/while/TensorArrayWrite/TensorArrayWriteV3/EnterEntermap/TensorArray_1*
T0*&
_class
loc:@map/while/DecodePng*
parallel_iterations
*
is_constant(*'

frame_namemap/while/while_context*
_output_shapes
:
h
map/while/add_1/yConst^map/while/Identity*
value	B :*
dtype0*
_output_shapes
: 
`
map/while/add_1Addmap/while/Identity_1map/while/add_1/y*
T0*
_output_shapes
: 
X
map/while/NextIterationNextIterationmap/while/add*
T0*
_output_shapes
: 
\
map/while/NextIteration_1NextIterationmap/while/add_1*
_output_shapes
: *
T0
z
map/while/NextIteration_2NextIteration-map/while/TensorArrayWrite/TensorArrayWriteV3*
T0*
_output_shapes
: 
I
map/while/ExitExitmap/while/Switch*
T0*
_output_shapes
: 
M
map/while/Exit_1Exitmap/while/Switch_1*
T0*
_output_shapes
: 
M
map/while/Exit_2Exitmap/while/Switch_2*
_output_shapes
: *
T0

&map/TensorArrayStack/TensorArraySizeV3TensorArraySizeV3map/TensorArray_1map/while/Exit_2*$
_class
loc:@map/TensorArray_1*
_output_shapes
: 

 map/TensorArrayStack/range/startConst*
value	B : *$
_class
loc:@map/TensorArray_1*
dtype0*
_output_shapes
: 

 map/TensorArrayStack/range/deltaConst*
value	B :*$
_class
loc:@map/TensorArray_1*
dtype0*
_output_shapes
: 
ζ
map/TensorArrayStack/rangeRange map/TensorArrayStack/range/start&map/TensorArrayStack/TensorArraySizeV3 map/TensorArrayStack/range/delta*$
_class
loc:@map/TensorArray_1*#
_output_shapes
:?????????*

Tidx0
©
(map/TensorArrayStack/TensorArrayGatherV3TensorArrayGatherV3map/TensorArray_1map/TensorArrayStack/rangemap/while/Exit_2*1
element_shape :??????????????????*$
_class
loc:@map/TensorArray_1*
dtype0*A
_output_shapes/
-:+???????????????????????????
‘
CastCast(map/TensorArrayStack/TensorArrayGatherV3*

SrcT0*
Truncate( *A
_output_shapes/
-:+???????????????????????????*

DstT0
^
Reshape/shapeConst*
valueB"????  *
dtype0*
_output_shapes
:
h
ReshapeReshapeCastReshape/shape*
T0*
Tshape0*(
_output_shapes
:?????????
I
xIdentityReshape*
T0*(
_output_shapes
:?????????
n
PlaceholderPlaceholder*
dtype0*'
_output_shapes
:?????????
*
shape:?????????

d
random_normal/shapeConst*
dtype0*
_output_shapes
:*
valueB"  
   
W
random_normal/meanConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Y
random_normal/stddevConst*
valueB
 *
Χ#<*
dtype0*
_output_shapes
: 

"random_normal/RandomStandardNormalRandomStandardNormalrandom_normal/shape*

seed *
T0*
dtype0*
_output_shapes
:	
*
seed2 
|
random_normal/mulMul"random_normal/RandomStandardNormalrandom_normal/stddev*
T0*
_output_shapes
:	

e
random_normalAddrandom_normal/mulrandom_normal/mean*
_output_shapes
:	
*
T0
x
W1
VariableV2*
dtype0*
_output_shapes
:	
*
	container *
shape:	
*
shared_name 

	W1/AssignAssignW1random_normal*
use_locking(*
T0*
_class
	loc:@W1*
validate_shape(*
_output_shapes
:	

X
W1/readIdentityW1*
_output_shapes
:	
*
T0*
_class
	loc:@W1
_
random_normal_1/shapeConst*
valueB:
*
dtype0*
_output_shapes
:
Y
random_normal_1/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    
[
random_normal_1/stddevConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 

$random_normal_1/RandomStandardNormalRandomStandardNormalrandom_normal_1/shape*

seed *
T0*
dtype0*
_output_shapes
:
*
seed2 
}
random_normal_1/mulMul$random_normal_1/RandomStandardNormalrandom_normal_1/stddev*
_output_shapes
:
*
T0
f
random_normal_1Addrandom_normal_1/mulrandom_normal_1/mean*
T0*
_output_shapes
:

n
B1
VariableV2*
dtype0*
_output_shapes
:
*
	container *
shape:
*
shared_name 

	B1/AssignAssignB1random_normal_1*
T0*
_class
	loc:@B1*
validate_shape(*
_output_shapes
:
*
use_locking(
S
B1/readIdentityB1*
_output_shapes
:
*
T0*
_class
	loc:@B1
t
MatMulMatMulxW1/read*'
_output_shapes
:?????????
*
transpose_a( *
transpose_b( *
T0
M
addAddMatMulB1/read*'
_output_shapes
:?????????
*
T0
I
SoftmaxSoftmaxadd*
T0*'
_output_shapes
:?????????

[
ArgMax/dimensionConst*
valueB :
?????????*
dtype0*
_output_shapes
: 
t
ArgMaxArgMaxaddArgMax/dimension*
T0*
output_type0	*#
_output_shapes
:?????????*

Tidx0
h
&softmax_cross_entropy_with_logits/RankConst*
value	B :*
dtype0*
_output_shapes
: 
j
'softmax_cross_entropy_with_logits/ShapeShapeadd*
T0*
out_type0*
_output_shapes
:
j
(softmax_cross_entropy_with_logits/Rank_1Const*
value	B :*
dtype0*
_output_shapes
: 
l
)softmax_cross_entropy_with_logits/Shape_1Shapeadd*
T0*
out_type0*
_output_shapes
:
i
'softmax_cross_entropy_with_logits/Sub/yConst*
value	B :*
dtype0*
_output_shapes
: 
 
%softmax_cross_entropy_with_logits/SubSub(softmax_cross_entropy_with_logits/Rank_1'softmax_cross_entropy_with_logits/Sub/y*
_output_shapes
: *
T0

-softmax_cross_entropy_with_logits/Slice/beginPack%softmax_cross_entropy_with_logits/Sub*
T0*

axis *
N*
_output_shapes
:
v
,softmax_cross_entropy_with_logits/Slice/sizeConst*
dtype0*
_output_shapes
:*
valueB:
κ
'softmax_cross_entropy_with_logits/SliceSlice)softmax_cross_entropy_with_logits/Shape_1-softmax_cross_entropy_with_logits/Slice/begin,softmax_cross_entropy_with_logits/Slice/size*
Index0*
T0*
_output_shapes
:

1softmax_cross_entropy_with_logits/concat/values_0Const*
valueB:
?????????*
dtype0*
_output_shapes
:
o
-softmax_cross_entropy_with_logits/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
ω
(softmax_cross_entropy_with_logits/concatConcatV21softmax_cross_entropy_with_logits/concat/values_0'softmax_cross_entropy_with_logits/Slice-softmax_cross_entropy_with_logits/concat/axis*
N*
_output_shapes
:*

Tidx0*
T0
¬
)softmax_cross_entropy_with_logits/ReshapeReshapeadd(softmax_cross_entropy_with_logits/concat*
T0*
Tshape0*0
_output_shapes
:??????????????????
j
(softmax_cross_entropy_with_logits/Rank_2Const*
value	B :*
dtype0*
_output_shapes
: 
t
)softmax_cross_entropy_with_logits/Shape_2ShapePlaceholder*
T0*
out_type0*
_output_shapes
:
k
)softmax_cross_entropy_with_logits/Sub_1/yConst*
value	B :*
dtype0*
_output_shapes
: 
€
'softmax_cross_entropy_with_logits/Sub_1Sub(softmax_cross_entropy_with_logits/Rank_2)softmax_cross_entropy_with_logits/Sub_1/y*
T0*
_output_shapes
: 

/softmax_cross_entropy_with_logits/Slice_1/beginPack'softmax_cross_entropy_with_logits/Sub_1*
N*
_output_shapes
:*
T0*

axis 
x
.softmax_cross_entropy_with_logits/Slice_1/sizeConst*
valueB:*
dtype0*
_output_shapes
:
π
)softmax_cross_entropy_with_logits/Slice_1Slice)softmax_cross_entropy_with_logits/Shape_2/softmax_cross_entropy_with_logits/Slice_1/begin.softmax_cross_entropy_with_logits/Slice_1/size*
Index0*
T0*
_output_shapes
:

3softmax_cross_entropy_with_logits/concat_1/values_0Const*
valueB:
?????????*
dtype0*
_output_shapes
:
q
/softmax_cross_entropy_with_logits/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 

*softmax_cross_entropy_with_logits/concat_1ConcatV23softmax_cross_entropy_with_logits/concat_1/values_0)softmax_cross_entropy_with_logits/Slice_1/softmax_cross_entropy_with_logits/concat_1/axis*
T0*
N*
_output_shapes
:*

Tidx0
Έ
+softmax_cross_entropy_with_logits/Reshape_1ReshapePlaceholder*softmax_cross_entropy_with_logits/concat_1*
T0*
Tshape0*0
_output_shapes
:??????????????????
δ
!softmax_cross_entropy_with_logitsSoftmaxCrossEntropyWithLogits)softmax_cross_entropy_with_logits/Reshape+softmax_cross_entropy_with_logits/Reshape_1*
T0*?
_output_shapes-
+:?????????:??????????????????
k
)softmax_cross_entropy_with_logits/Sub_2/yConst*
value	B :*
dtype0*
_output_shapes
: 
’
'softmax_cross_entropy_with_logits/Sub_2Sub&softmax_cross_entropy_with_logits/Rank)softmax_cross_entropy_with_logits/Sub_2/y*
T0*
_output_shapes
: 
y
/softmax_cross_entropy_with_logits/Slice_2/beginConst*
dtype0*
_output_shapes
:*
valueB: 

.softmax_cross_entropy_with_logits/Slice_2/sizePack'softmax_cross_entropy_with_logits/Sub_2*
N*
_output_shapes
:*
T0*

axis 
ξ
)softmax_cross_entropy_with_logits/Slice_2Slice'softmax_cross_entropy_with_logits/Shape/softmax_cross_entropy_with_logits/Slice_2/begin.softmax_cross_entropy_with_logits/Slice_2/size*
_output_shapes
:*
Index0*
T0
ΐ
+softmax_cross_entropy_with_logits/Reshape_2Reshape!softmax_cross_entropy_with_logits)softmax_cross_entropy_with_logits/Slice_2*
T0*
Tshape0*#
_output_shapes
:?????????
O
ConstConst*
valueB: *
dtype0*
_output_shapes
:
~
MeanMean+softmax_cross_entropy_with_logits/Reshape_2Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
R
gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
X
gradients/grad_ys_0Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 
o
gradients/FillFillgradients/Shapegradients/grad_ys_0*
T0*

index_type0*
_output_shapes
: 
k
!gradients/Mean_grad/Reshape/shapeConst*
valueB:*
dtype0*
_output_shapes
:

gradients/Mean_grad/ReshapeReshapegradients/Fill!gradients/Mean_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
:

gradients/Mean_grad/ShapeShape+softmax_cross_entropy_with_logits/Reshape_2*
T0*
out_type0*
_output_shapes
:

gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/Shape*#
_output_shapes
:?????????*

Tmultiples0*
T0

gradients/Mean_grad/Shape_1Shape+softmax_cross_entropy_with_logits/Reshape_2*
T0*
out_type0*
_output_shapes
:
^
gradients/Mean_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
c
gradients/Mean_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:

gradients/Mean_grad/ProdProdgradients/Mean_grad/Shape_1gradients/Mean_grad/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
e
gradients/Mean_grad/Const_1Const*
valueB: *
dtype0*
_output_shapes
:

gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_2gradients/Mean_grad/Const_1*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
_
gradients/Mean_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 

gradients/Mean_grad/MaximumMaximumgradients/Mean_grad/Prod_1gradients/Mean_grad/Maximum/y*
T0*
_output_shapes
: 

gradients/Mean_grad/floordivFloorDivgradients/Mean_grad/Prodgradients/Mean_grad/Maximum*
T0*
_output_shapes
: 
~
gradients/Mean_grad/CastCastgradients/Mean_grad/floordiv*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0

gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Cast*#
_output_shapes
:?????????*
T0
‘
@gradients/softmax_cross_entropy_with_logits/Reshape_2_grad/ShapeShape!softmax_cross_entropy_with_logits*
T0*
out_type0*
_output_shapes
:
θ
Bgradients/softmax_cross_entropy_with_logits/Reshape_2_grad/ReshapeReshapegradients/Mean_grad/truediv@gradients/softmax_cross_entropy_with_logits/Reshape_2_grad/Shape*
T0*
Tshape0*#
_output_shapes
:?????????

gradients/zeros_like	ZerosLike#softmax_cross_entropy_with_logits:1*
T0*0
_output_shapes
:??????????????????

?gradients/softmax_cross_entropy_with_logits_grad/ExpandDims/dimConst*
valueB :
?????????*
dtype0*
_output_shapes
: 

;gradients/softmax_cross_entropy_with_logits_grad/ExpandDims
ExpandDimsBgradients/softmax_cross_entropy_with_logits/Reshape_2_grad/Reshape?gradients/softmax_cross_entropy_with_logits_grad/ExpandDims/dim*
T0*'
_output_shapes
:?????????*

Tdim0
Ψ
4gradients/softmax_cross_entropy_with_logits_grad/mulMul;gradients/softmax_cross_entropy_with_logits_grad/ExpandDims#softmax_cross_entropy_with_logits:1*
T0*0
_output_shapes
:??????????????????
―
;gradients/softmax_cross_entropy_with_logits_grad/LogSoftmax
LogSoftmax)softmax_cross_entropy_with_logits/Reshape*
T0*0
_output_shapes
:??????????????????
³
4gradients/softmax_cross_entropy_with_logits_grad/NegNeg;gradients/softmax_cross_entropy_with_logits_grad/LogSoftmax*
T0*0
_output_shapes
:??????????????????

Agradients/softmax_cross_entropy_with_logits_grad/ExpandDims_1/dimConst*
valueB :
?????????*
dtype0*
_output_shapes
: 

=gradients/softmax_cross_entropy_with_logits_grad/ExpandDims_1
ExpandDimsBgradients/softmax_cross_entropy_with_logits/Reshape_2_grad/ReshapeAgradients/softmax_cross_entropy_with_logits_grad/ExpandDims_1/dim*

Tdim0*
T0*'
_output_shapes
:?????????
ν
6gradients/softmax_cross_entropy_with_logits_grad/mul_1Mul=gradients/softmax_cross_entropy_with_logits_grad/ExpandDims_14gradients/softmax_cross_entropy_with_logits_grad/Neg*
T0*0
_output_shapes
:??????????????????
Ή
Agradients/softmax_cross_entropy_with_logits_grad/tuple/group_depsNoOp5^gradients/softmax_cross_entropy_with_logits_grad/mul7^gradients/softmax_cross_entropy_with_logits_grad/mul_1
Σ
Igradients/softmax_cross_entropy_with_logits_grad/tuple/control_dependencyIdentity4gradients/softmax_cross_entropy_with_logits_grad/mulB^gradients/softmax_cross_entropy_with_logits_grad/tuple/group_deps*0
_output_shapes
:??????????????????*
T0*G
_class=
;9loc:@gradients/softmax_cross_entropy_with_logits_grad/mul
Ω
Kgradients/softmax_cross_entropy_with_logits_grad/tuple/control_dependency_1Identity6gradients/softmax_cross_entropy_with_logits_grad/mul_1B^gradients/softmax_cross_entropy_with_logits_grad/tuple/group_deps*0
_output_shapes
:??????????????????*
T0*I
_class?
=;loc:@gradients/softmax_cross_entropy_with_logits_grad/mul_1

>gradients/softmax_cross_entropy_with_logits/Reshape_grad/ShapeShapeadd*
T0*
out_type0*
_output_shapes
:

@gradients/softmax_cross_entropy_with_logits/Reshape_grad/ReshapeReshapeIgradients/softmax_cross_entropy_with_logits_grad/tuple/control_dependency>gradients/softmax_cross_entropy_with_logits/Reshape_grad/Shape*
T0*
Tshape0*'
_output_shapes
:?????????

^
gradients/add_grad/ShapeShapeMatMul*
_output_shapes
:*
T0*
out_type0
d
gradients/add_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:

΄
(gradients/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_grad/Shapegradients/add_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
Ι
gradients/add_grad/SumSum@gradients/softmax_cross_entropy_with_logits/Reshape_grad/Reshape(gradients/add_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0

gradients/add_grad/ReshapeReshapegradients/add_grad/Sumgradients/add_grad/Shape*
T0*
Tshape0*'
_output_shapes
:?????????

Ν
gradients/add_grad/Sum_1Sum@gradients/softmax_cross_entropy_with_logits/Reshape_grad/Reshape*gradients/add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0

gradients/add_grad/Reshape_1Reshapegradients/add_grad/Sum_1gradients/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:

g
#gradients/add_grad/tuple/group_depsNoOp^gradients/add_grad/Reshape^gradients/add_grad/Reshape_1
Ϊ
+gradients/add_grad/tuple/control_dependencyIdentitygradients/add_grad/Reshape$^gradients/add_grad/tuple/group_deps*'
_output_shapes
:?????????
*
T0*-
_class#
!loc:@gradients/add_grad/Reshape
Σ
-gradients/add_grad/tuple/control_dependency_1Identitygradients/add_grad/Reshape_1$^gradients/add_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/add_grad/Reshape_1*
_output_shapes
:

΅
gradients/MatMul_grad/MatMulMatMul+gradients/add_grad/tuple/control_dependencyW1/read*
T0*(
_output_shapes
:?????????*
transpose_a( *
transpose_b(
¨
gradients/MatMul_grad/MatMul_1MatMulx+gradients/add_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes
:	
*
transpose_a(
n
&gradients/MatMul_grad/tuple/group_depsNoOp^gradients/MatMul_grad/MatMul^gradients/MatMul_grad/MatMul_1
ε
.gradients/MatMul_grad/tuple/control_dependencyIdentitygradients/MatMul_grad/MatMul'^gradients/MatMul_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/MatMul_grad/MatMul*(
_output_shapes
:?????????
β
0gradients/MatMul_grad/tuple/control_dependency_1Identitygradients/MatMul_grad/MatMul_1'^gradients/MatMul_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/MatMul_grad/MatMul_1*
_output_shapes
:	

u
beta1_power/initial_valueConst*
valueB
 *fff?*
_class
	loc:@B1*
dtype0*
_output_shapes
: 

beta1_power
VariableV2*
shared_name *
_class
	loc:@B1*
	container *
shape: *
dtype0*
_output_shapes
: 
₯
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
use_locking(*
T0*
_class
	loc:@B1*
validate_shape(*
_output_shapes
: 
a
beta1_power/readIdentitybeta1_power*
_output_shapes
: *
T0*
_class
	loc:@B1
u
beta2_power/initial_valueConst*
dtype0*
_output_shapes
: *
valueB
 *wΎ?*
_class
	loc:@B1

beta2_power
VariableV2*
shared_name *
_class
	loc:@B1*
	container *
shape: *
dtype0*
_output_shapes
: 
₯
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
T0*
_class
	loc:@B1*
validate_shape(*
_output_shapes
: *
use_locking(
a
beta2_power/readIdentitybeta2_power*
T0*
_class
	loc:@B1*
_output_shapes
: 

)W1/Adam/Initializer/zeros/shape_as_tensorConst*
_class
	loc:@W1*
valueB"  
   *
dtype0*
_output_shapes
:
{
W1/Adam/Initializer/zeros/ConstConst*
_class
	loc:@W1*
valueB
 *    *
dtype0*
_output_shapes
: 
ΐ
W1/Adam/Initializer/zerosFill)W1/Adam/Initializer/zeros/shape_as_tensorW1/Adam/Initializer/zeros/Const*
T0*
_class
	loc:@W1*

index_type0*
_output_shapes
:	


W1/Adam
VariableV2*
shape:	
*
dtype0*
_output_shapes
:	
*
shared_name *
_class
	loc:@W1*
	container 
¦
W1/Adam/AssignAssignW1/AdamW1/Adam/Initializer/zeros*
use_locking(*
T0*
_class
	loc:@W1*
validate_shape(*
_output_shapes
:	

b
W1/Adam/readIdentityW1/Adam*
T0*
_class
	loc:@W1*
_output_shapes
:	


+W1/Adam_1/Initializer/zeros/shape_as_tensorConst*
_class
	loc:@W1*
valueB"  
   *
dtype0*
_output_shapes
:
}
!W1/Adam_1/Initializer/zeros/ConstConst*
_class
	loc:@W1*
valueB
 *    *
dtype0*
_output_shapes
: 
Ζ
W1/Adam_1/Initializer/zerosFill+W1/Adam_1/Initializer/zeros/shape_as_tensor!W1/Adam_1/Initializer/zeros/Const*
T0*
_class
	loc:@W1*

index_type0*
_output_shapes
:	


	W1/Adam_1
VariableV2*
_class
	loc:@W1*
	container *
shape:	
*
dtype0*
_output_shapes
:	
*
shared_name 
¬
W1/Adam_1/AssignAssign	W1/Adam_1W1/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
	loc:@W1*
validate_shape(*
_output_shapes
:	

f
W1/Adam_1/readIdentity	W1/Adam_1*
T0*
_class
	loc:@W1*
_output_shapes
:	

}
B1/Adam/Initializer/zerosConst*
dtype0*
_output_shapes
:
*
_class
	loc:@B1*
valueB
*    

B1/Adam
VariableV2*
dtype0*
_output_shapes
:
*
shared_name *
_class
	loc:@B1*
	container *
shape:

‘
B1/Adam/AssignAssignB1/AdamB1/Adam/Initializer/zeros*
validate_shape(*
_output_shapes
:
*
use_locking(*
T0*
_class
	loc:@B1
]
B1/Adam/readIdentityB1/Adam*
T0*
_class
	loc:@B1*
_output_shapes
:


B1/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes
:
*
_class
	loc:@B1*
valueB
*    

	B1/Adam_1
VariableV2*
shape:
*
dtype0*
_output_shapes
:
*
shared_name *
_class
	loc:@B1*
	container 
§
B1/Adam_1/AssignAssign	B1/Adam_1B1/Adam_1/Initializer/zeros*
T0*
_class
	loc:@B1*
validate_shape(*
_output_shapes
:
*
use_locking(
a
B1/Adam_1/readIdentity	B1/Adam_1*
T0*
_class
	loc:@B1*
_output_shapes
:

W
Adam/learning_rateConst*
valueB
 *ΝΜL=*
dtype0*
_output_shapes
: 
O

Adam/beta1Const*
valueB
 *fff?*
dtype0*
_output_shapes
: 
O

Adam/beta2Const*
valueB
 *wΎ?*
dtype0*
_output_shapes
: 
Q
Adam/epsilonConst*
dtype0*
_output_shapes
: *
valueB
 *wΜ+2
΅
Adam/update_W1/ApplyAdam	ApplyAdamW1W1/Adam	W1/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon0gradients/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
	loc:@W1*
use_nesterov( *
_output_shapes
:	

­
Adam/update_B1/ApplyAdam	ApplyAdamB1B1/Adam	B1/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon-gradients/add_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
	loc:@B1*
use_nesterov( *
_output_shapes
:


Adam/mulMulbeta1_power/read
Adam/beta1^Adam/update_B1/ApplyAdam^Adam/update_W1/ApplyAdam*
T0*
_class
	loc:@B1*
_output_shapes
: 

Adam/AssignAssignbeta1_powerAdam/mul*
use_locking( *
T0*
_class
	loc:@B1*
validate_shape(*
_output_shapes
: 


Adam/mul_1Mulbeta2_power/read
Adam/beta2^Adam/update_B1/ApplyAdam^Adam/update_W1/ApplyAdam*
T0*
_class
	loc:@B1*
_output_shapes
: 

Adam/Assign_1Assignbeta2_power
Adam/mul_1*
validate_shape(*
_output_shapes
: *
use_locking( *
T0*
_class
	loc:@B1
`
AdamNoOp^Adam/Assign^Adam/Assign_1^Adam/update_B1/ApplyAdam^Adam/update_W1/ApplyAdam
T
ArgMax_1/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 
x
ArgMax_1ArgMaxaddArgMax_1/dimension*

Tidx0*
T0*
output_type0	*#
_output_shapes
:?????????
T
ArgMax_2/dimensionConst*
dtype0*
_output_shapes
: *
value	B :

ArgMax_2ArgMaxPlaceholderArgMax_2/dimension*
T0*
output_type0	*#
_output_shapes
:?????????*

Tidx0
P
EqualEqualArgMax_1ArgMax_2*
T0	*#
_output_shapes
:?????????
b
Cast_1CastEqual*
Truncate( *#
_output_shapes
:?????????*

DstT0*

SrcT0

Q
Const_1Const*
valueB: *
dtype0*
_output_shapes
:
]
Mean_1MeanCast_1Const_1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
Y
save/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
n
save/filenamePlaceholderWithDefaultsave/filename/input*
shape: *
dtype0*
_output_shapes
: 
e

save/ConstPlaceholderWithDefaultsave/filename*
dtype0*
_output_shapes
: *
shape: 
©
save/SaveV2/tensor_namesConst*
dtype0*
_output_shapes
:*]
valueTBRBB1BB1/AdamB	B1/Adam_1BW1BW1/AdamB	W1/Adam_1Bbeta1_powerBbeta2_power
s
save/SaveV2/shape_and_slicesConst*#
valueBB B B B B B B B *
dtype0*
_output_shapes
:
»
save/SaveV2SaveV2
save/Constsave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesB1B1/Adam	B1/Adam_1W1W1/Adam	W1/Adam_1beta1_powerbeta2_power*
dtypes

2
}
save/control_dependencyIdentity
save/Const^save/SaveV2*
_output_shapes
: *
T0*
_class
loc:@save/Const
»
save/RestoreV2/tensor_namesConst"/device:CPU:0*]
valueTBRBB1BB1/AdamB	B1/Adam_1BW1BW1/AdamB	W1/Adam_1Bbeta1_powerBbeta2_power*
dtype0*
_output_shapes
:

save/RestoreV2/shape_and_slicesConst"/device:CPU:0*#
valueBB B B B B B B B *
dtype0*
_output_shapes
:
Β
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*4
_output_shapes"
 ::::::::*
dtypes

2

save/AssignAssignB1save/RestoreV2*
use_locking(*
T0*
_class
	loc:@B1*
validate_shape(*
_output_shapes
:


save/Assign_1AssignB1/Adamsave/RestoreV2:1*
use_locking(*
T0*
_class
	loc:@B1*
validate_shape(*
_output_shapes
:


save/Assign_2Assign	B1/Adam_1save/RestoreV2:2*
use_locking(*
T0*
_class
	loc:@B1*
validate_shape(*
_output_shapes
:


save/Assign_3AssignW1save/RestoreV2:3*
use_locking(*
T0*
_class
	loc:@W1*
validate_shape(*
_output_shapes
:	


save/Assign_4AssignW1/Adamsave/RestoreV2:4*
validate_shape(*
_output_shapes
:	
*
use_locking(*
T0*
_class
	loc:@W1

save/Assign_5Assign	W1/Adam_1save/RestoreV2:5*
validate_shape(*
_output_shapes
:	
*
use_locking(*
T0*
_class
	loc:@W1

save/Assign_6Assignbeta1_powersave/RestoreV2:6*
use_locking(*
T0*
_class
	loc:@B1*
validate_shape(*
_output_shapes
: 

save/Assign_7Assignbeta2_powersave/RestoreV2:7*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
	loc:@B1

save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7
[
save_1/filename/inputConst*
dtype0*
_output_shapes
: *
valueB Bmodel
r
save_1/filenamePlaceholderWithDefaultsave_1/filename/input*
dtype0*
_output_shapes
: *
shape: 
i
save_1/ConstPlaceholderWithDefaultsave_1/filename*
dtype0*
_output_shapes
: *
shape: 

save_1/StringJoin/inputs_1Const*<
value3B1 B+_temp_64fe93114ce54cf39ee0c9263e3eccb7/part*
dtype0*
_output_shapes
: 
{
save_1/StringJoin
StringJoinsave_1/Constsave_1/StringJoin/inputs_1*
	separator *
N*
_output_shapes
: 
S
save_1/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
m
save_1/ShardedFilename/shardConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 

save_1/ShardedFilenameShardedFilenamesave_1/StringJoinsave_1/ShardedFilename/shardsave_1/num_shards"/device:CPU:0*
_output_shapes
: 
Ί
save_1/SaveV2/tensor_namesConst"/device:CPU:0*]
valueTBRBB1BB1/AdamB	B1/Adam_1BW1BW1/AdamB	W1/Adam_1Bbeta1_powerBbeta2_power*
dtype0*
_output_shapes
:

save_1/SaveV2/shape_and_slicesConst"/device:CPU:0*#
valueBB B B B B B B B *
dtype0*
_output_shapes
:
ά
save_1/SaveV2SaveV2save_1/ShardedFilenamesave_1/SaveV2/tensor_namessave_1/SaveV2/shape_and_slicesB1B1/Adam	B1/Adam_1W1W1/Adam	W1/Adam_1beta1_powerbeta2_power"/device:CPU:0*
dtypes

2
¨
save_1/control_dependencyIdentitysave_1/ShardedFilename^save_1/SaveV2"/device:CPU:0*
T0*)
_class
loc:@save_1/ShardedFilename*
_output_shapes
: 
²
-save_1/MergeV2Checkpoints/checkpoint_prefixesPacksave_1/ShardedFilename^save_1/control_dependency"/device:CPU:0*
T0*

axis *
N*
_output_shapes
:

save_1/MergeV2CheckpointsMergeV2Checkpoints-save_1/MergeV2Checkpoints/checkpoint_prefixessave_1/Const"/device:CPU:0*
delete_old_dirs(

save_1/IdentityIdentitysave_1/Const^save_1/MergeV2Checkpoints^save_1/control_dependency"/device:CPU:0*
T0*
_output_shapes
: 
½
save_1/RestoreV2/tensor_namesConst"/device:CPU:0*]
valueTBRBB1BB1/AdamB	B1/Adam_1BW1BW1/AdamB	W1/Adam_1Bbeta1_powerBbeta2_power*
dtype0*
_output_shapes
:

!save_1/RestoreV2/shape_and_slicesConst"/device:CPU:0*#
valueBB B B B B B B B *
dtype0*
_output_shapes
:
Κ
save_1/RestoreV2	RestoreV2save_1/Constsave_1/RestoreV2/tensor_names!save_1/RestoreV2/shape_and_slices"/device:CPU:0*4
_output_shapes"
 ::::::::*
dtypes

2

save_1/AssignAssignB1save_1/RestoreV2*
T0*
_class
	loc:@B1*
validate_shape(*
_output_shapes
:
*
use_locking(

save_1/Assign_1AssignB1/Adamsave_1/RestoreV2:1*
use_locking(*
T0*
_class
	loc:@B1*
validate_shape(*
_output_shapes
:


save_1/Assign_2Assign	B1/Adam_1save_1/RestoreV2:2*
use_locking(*
T0*
_class
	loc:@B1*
validate_shape(*
_output_shapes
:


save_1/Assign_3AssignW1save_1/RestoreV2:3*
use_locking(*
T0*
_class
	loc:@W1*
validate_shape(*
_output_shapes
:	

 
save_1/Assign_4AssignW1/Adamsave_1/RestoreV2:4*
T0*
_class
	loc:@W1*
validate_shape(*
_output_shapes
:	
*
use_locking(
’
save_1/Assign_5Assign	W1/Adam_1save_1/RestoreV2:5*
T0*
_class
	loc:@W1*
validate_shape(*
_output_shapes
:	
*
use_locking(

save_1/Assign_6Assignbeta1_powersave_1/RestoreV2:6*
T0*
_class
	loc:@B1*
validate_shape(*
_output_shapes
: *
use_locking(

save_1/Assign_7Assignbeta2_powersave_1/RestoreV2:7*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
	loc:@B1
ͺ
save_1/restore_shardNoOp^save_1/Assign^save_1/Assign_1^save_1/Assign_2^save_1/Assign_3^save_1/Assign_4^save_1/Assign_5^save_1/Assign_6^save_1/Assign_7
1
save_1/restore_allNoOp^save_1/restore_shard"&B
save_1/Const:0save_1/Identity:0save_1/restore_all (5 @F8"}
trainable_variablesfd
/
W1:0	W1/Assign	W1/read:02random_normal:08
1
B1:0	B1/Assign	B1/read:02random_normal_1:08"
train_op

Adam"?
while_contextΐ½
Ί
map/while/while_context
*map/while/LoopCond:02map/while/Merge:0:map/while/Identity:0Bmap/while/Exit:0Bmap/while/Exit_1:0Bmap/while/Exit_2:0JΟ

map/TensorArray:0
@map/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0
map/TensorArray_1:0
map/strided_slice:0
map/while/DecodePng:0
map/while/Enter:0
map/while/Enter_1:0
map/while/Enter_2:0
map/while/Exit:0
map/while/Exit_1:0
map/while/Exit_2:0
map/while/Identity:0
map/while/Identity_1:0
map/while/Identity_2:0
map/while/Less/Enter:0
map/while/Less:0
map/while/Less_1:0
map/while/LogicalAnd:0
map/while/LoopCond:0
map/while/Merge:0
map/while/Merge:1
map/while/Merge_1:0
map/while/Merge_1:1
map/while/Merge_2:0
map/while/Merge_2:1
map/while/NextIteration:0
map/while/NextIteration_1:0
map/while/NextIteration_2:0
map/while/Switch:0
map/while/Switch:1
map/while/Switch_1:0
map/while/Switch_1:1
map/while/Switch_2:0
map/while/Switch_2:1
#map/while/TensorArrayReadV3/Enter:0
%map/while/TensorArrayReadV3/Enter_1:0
map/while/TensorArrayReadV3:0
5map/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0
/map/while/TensorArrayWrite/TensorArrayWriteV3:0
map/while/add/y:0
map/while/add:0
map/while/add_1/y:0
map/while/add_1:0-
map/strided_slice:0map/while/Less/Enter:08
map/TensorArray:0#map/while/TensorArrayReadV3/Enter:0i
@map/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0%map/while/TensorArrayReadV3/Enter_1:0L
map/TensorArray_1:05map/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0Rmap/while/Enter:0Rmap/while/Enter_1:0Rmap/while/Enter_2:0Zmap/strided_slice:0"Ω
	variablesΛΘ
/
W1:0	W1/Assign	W1/read:02random_normal:08
1
B1:0	B1/Assign	B1/read:02random_normal_1:08
T
beta1_power:0beta1_power/Assignbeta1_power/read:02beta1_power/initial_value:0
T
beta2_power:0beta2_power/Assignbeta2_power/read:02beta2_power/initial_value:0
H
	W1/Adam:0W1/Adam/AssignW1/Adam/read:02W1/Adam/Initializer/zeros:0
P
W1/Adam_1:0W1/Adam_1/AssignW1/Adam_1/read:02W1/Adam_1/Initializer/zeros:0
H
	B1/Adam:0B1/Adam/AssignB1/Adam/read:02B1/Adam/Initializer/zeros:0
P
B1/Adam_1:0B1/Adam_1/AssignB1/Adam_1/read:02B1/Adam_1/Initializer/zeros:0*?
serving_default
(

raw_inputs
raw_x:0?????????&
outputs
ArgMax:0	?????????*
scores 
	Softmax:0?????????
tensorflow/serving/predict