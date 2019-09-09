// 2019/09/01 Hiroyuki Ogasawara
// vim:ts=4 sw=4 noet:

#include	<tensorflow/c/c_api.h>
#include	<string.h>
#include	"CCTFLibrary.h"
#include	"CCTFSystem.h"


namespace cctf {
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

#define	CCTF_CHECK_STATUS( status )	CCTF_CheckStatus( status, __func__, __LINE__ )

static bool	CCTF_CheckStatus( CCStatus& status, const char* text, int line )
{
	if( status.IsError() ){
		CC_ERROR( "CCTF API Error %d %s : %s\n", line, text, TF_Message( status.IPointer() ) );
		return	false;
	}
	return	true;
}


//-----------------------------------------------------------------------------

#define	DATA_TYPE_DEF( name )	{	#name, TF_##name,	}
const char*	GetTypeString( TF_DataType type )
{
	struct DataTypeString {
		const char*	TypeName;
		TF_DataType	Type;
	};
	static DataTypeString	type_list[]= {
		{	"Unknown",	(TF_DataType)0,	},
		DATA_TYPE_DEF( FLOAT ),
		DATA_TYPE_DEF( DOUBLE ),
		DATA_TYPE_DEF( INT32 ),
		DATA_TYPE_DEF( UINT32 ),
		DATA_TYPE_DEF( INT16 ),
		DATA_TYPE_DEF( INT8 ),
		DATA_TYPE_DEF( STRING ),
		DATA_TYPE_DEF( COMPLEX ),
		DATA_TYPE_DEF( INT64 ),
		DATA_TYPE_DEF( BOOL ),
		DATA_TYPE_DEF( QINT8 ),
		DATA_TYPE_DEF( QUINT8 ),
		DATA_TYPE_DEF( QINT32 ),
		DATA_TYPE_DEF( BFLOAT16 ),
		DATA_TYPE_DEF( QINT16 ),
		DATA_TYPE_DEF( QUINT16 ),
		DATA_TYPE_DEF( UINT16 ),
		DATA_TYPE_DEF( COMPLEX128 ),
		DATA_TYPE_DEF( HALF ),
		DATA_TYPE_DEF( RESOURCE ),
		DATA_TYPE_DEF( VARIANT ),
		DATA_TYPE_DEF( UINT32 ),
		DATA_TYPE_DEF( UINT64 ),
	};
	unsigned int	table_size= sizeof(type_list)/sizeof(DataTypeString);
	unsigned int	index= (int)type;
	if( index < table_size ){
		const auto&	table= type_list[ (int)type ];
		assert( table.Type == type );
		return	table.TypeName;
	}
	return	"Error";
}



//-----------------------------------------------------------------------------

CCStatus::CCStatus()
{
	iPointer= TF_NewStatus();
}

CCStatus::~CCStatus()
{
	Finalize();
}

void	CCStatus::Finalize()
{
	if( iPointer ){
		TF_DeleteStatus( iPointer );
		iPointer= nullptr;
	}
}

bool	CCStatus::IsOK() const
{
	return	iPointer && TF_GetCode( iPointer ) == TF_OK;
}

bool	CCStatus::IsError() const
{
	return	iPointer && TF_GetCode( iPointer ) != TF_OK;
}


//-----------------------------------------------------------------------------

void	CCBuffer::cb_Deallocator( void* data, size_t size )
{
	CCSystem::Free( data, size );
}

CCBuffer::CCBuffer()
{
	SetPointer( TF_NewBuffer() );
}

CCBuffer::~CCBuffer()
{
	Finalize();
}

void	CCBuffer::Finalize()
{
	if( iPointer ){
		TF_DeleteBuffer( iPointer );
		iPointer= nullptr;
	}
}

#if 0
void	CCBuffer::Create( size_t size )
{
	Finalize();
	SetPointer( TF_NewBuffer() );
	iPointer->data= CCSystem::AllocByte<void>( size );
	iPointer->length= size;
	iPointer->data_deallocator= cb_Deallocator;
}
#endif

const void*	CCBuffer::GetData() const
{
	return	IPointer()->data;
}

size_t	CCBuffer::GetDataSize() const
{
	return	IPointer()->length;
}

bool	CCBuffer::LoadFile( const char* path )
{
	Finalize();

	size_t	size= 0;
	void*	buffer= CCSystem::LoadFile( path, size );
	if( !buffer ){
		return	false;
	}

	SetPointer( TF_NewBuffer() );

	iPointer->data= buffer;
	iPointer->length= size;
	iPointer->data_deallocator= cb_Deallocator;

	return	true;
}

bool	CCBuffer::SaveFile( const char* path )
{
	return	CCSystem::SaveFile( path, GetData(), GetDataSize() );
}


//-----------------------------------------------------------------------------

void	CCTensor::cb_Deallocator( void* data, size_t size, void* arg )
{
	CCSystem::Free( data, size );
}

CCTensor::CCTensor()
{
}

CCTensor::CCTensor( TF_DataType type, const CCShape& shape )
{
	Allocate( type, shape );
}

CCTensor::~CCTensor()
{
	Finalize();
}

void	CCTensor::Finalize()
{
	if( iPointer ){
		TF_DeleteTensor( iPointer );
		iPointer= nullptr;
	}
}

void	CCTensor::Allocate( TF_DataType type, const CCShape& shape )
{
	Finalize();
	int64_t	dims[CCShape::RANK_MAX];
	int	rank= shape.GetDims( dims, CCShape::RANK_MAX );
	SetPointer( TF_AllocateTensor( type, dims, rank, shape.GetElementCount() * TF_DataTypeSize( type ) ) );
}

void	CCTensor::SetData( TF_DataType type, const CCShape& shape, const void* data, size_t data_size )
{
	Finalize();
	auto*	buffer= CCSystem::AllocByte<void>( data_size );
	memcpy( buffer, data, data_size );
	int64_t	dims[CCShape::RANK_MAX];
	int	rank= shape.GetDims( dims, CCShape::RANK_MAX );
	SetPointer( TF_NewTensor( type, dims, rank, buffer, data_size, cb_Deallocator, this ) );
}

bool	CCTensor::SetString( const char* text )
{
	Finalize();
	constexpr int	STRING_OFFSET_SIZE= 8;
	size_t	str_length= strlen(text);
	size_t	byte_size= TF_StringEncodedSize( str_length );
	SetPointer( TF_AllocateTensor( TF_STRING, nullptr, 0, byte_size + STRING_OFFSET_SIZE ) );
	char*	buffer= Map<char>();
	memset( buffer, 0, STRING_OFFSET_SIZE );
	CCStatus	status;
	TF_StringEncode( text, str_length, buffer + STRING_OFFSET_SIZE, byte_size, status.IPointer() );
	if( !status.IsOK() ){
		Finalize();
	}
	return	status.IsOK();
}

TF_DataType	CCTensor::GetType() const
{
	return	TF_TensorType( IPointer() );
}

int64_t	CCTensor::GetElementCount() const
{
	return	TF_TensorElementCount( IPointer() );
}

size_t	CCTensor::GetByteSize() const
{
	return	TF_TensorByteSize( IPointer() );
}

void*	CCTensor::GetData() const
{
	return	TF_TensorData( IPointer() );
}

void	CCTensor::GetShape( CCShape& shape ) const
{
	unsigned int	rank= TF_NumDims( IPointer() );
	assert( rank < CCShape::RANK_MAX );
	shape.SetRank( rank );
	for( unsigned int ri= 0 ; ri< rank ; ri++ ){
		shape.Set( ri, static_cast<int>( TF_Dim( IPointer(), ri ) ) );
	}
}

static const char*	GetNest( unsigned int nest )
{
	const char*	space= "          ";
	assert( strlen(space) == 10 );
	return	space + 10 - nest;
}

static void	Dump_R1( const float* fptr, const CCShape& shape, unsigned int nest )
{
	const char*	nest_str= GetNest( nest );
	CC_PRINT( "%s[ ", nest_str );
	unsigned int	count= shape.GetX();
	for( unsigned int fi= 0 ; fi< count ; fi++ ){
		CC_PRINT( "%f, ", *fptr );
		fptr++;
	}
	CC_PRINT( "],\n" );
}

static void	Dump_R2( const float* fptr, const CCShape& shape, unsigned int nest )
{
	const char*	nest_str= GetNest( nest );
	CC_PRINT( "%s[\n", nest_str );
	unsigned int	count= shape.GetY();
	unsigned int	page_size= shape.GetX();
	for( unsigned int fi= 0 ; fi< count ; fi++ ){
		Dump_R1( fptr, shape, nest+1 );
		fptr+= page_size;
	}
	CC_PRINT( "%s],\n", nest_str );
}

static void	Dump_R3( const float* fptr, const CCShape& shape, unsigned int nest )
{
	const char*	nest_str= GetNest( nest );
	CC_PRINT( "%s[\n", nest_str );
	unsigned int	count= shape.GetZ();
	unsigned int	page_size= shape.GetX() * shape.GetY();
	for( unsigned int fi= 0 ; fi< count ; fi++ ){
		Dump_R2( fptr, shape, nest+1 );
		fptr+= page_size;
	}
	CC_PRINT( "%s],\n", nest_str );
}

static void	Dump_R4( const float* fptr, const CCShape& shape, unsigned int nest )
{
	const char*	nest_str= GetNest( nest );
	CC_PRINT( "%s[\n", nest_str );
	unsigned int	count= shape.GetW();
	unsigned int	page_size= shape.GetX() * shape.GetY() * shape.GetZ();
	for( unsigned int fi= 0 ; fi< count ; fi++ ){
		Dump_R3( fptr, shape, nest+1 );
		fptr+= page_size;
	}
	CC_PRINT( "%s],\n", nest_str );
}

void	CCTensor::Dump( const char* text ) const
{
	CCShape	shape;
	GetShape( shape );
	CC_PRINT( "Tensor %s type=%d count=%d byte=%d Shape:", text, GetType(), GetElementCount(), GetByteSize() );
	shape.Dump( "" );
	if( GetType() != TF_FLOAT ){
		return;
	}
	float*	fptr= Map<float>();
	switch( shape.GetRank() ){
	case 0: // scalar
		if( GetElementCount() >= 1 ){
			CC_PRINT( "%f\n", fptr[0] );
		}
		break;
	case 1: Dump_R1( fptr, shape, 0 ); break;
	case 2: Dump_R2( fptr, shape, 0 ); break;
	case 3: Dump_R3( fptr, shape, 0 ); break;
	case 4: Dump_R4( fptr, shape, 0 ); break;
	default:
		assert( 0 );
	}
}


//-----------------------------------------------------------------------------

CCOperation::CCOperation()
{
}

CCOperation::~CCOperation()
{
}

void	CCOperation::Finalize()
{
}

int		CCOperation::GetInputCount() const
{
	return	TF_OperationNumInputs( IPointer() );
}

int		CCOperation::GetOutputCount() const
{
	return	TF_OperationNumOutputs( IPointer() );
}

const char*	CCOperation::GetDevice() const
{
	return	TF_OperationDevice( IPointer() );
}

void	CCOperation::Dump_Input()
{
	int	count= TF_OperationNumInputs( IPointer() );
	for( int ci= 0 ; ci< count ; ci++ ){
		TF_DataType	type= TF_OperationInputType( { IPointer(), ci } );
		TF_Output	input= TF_OperationInput( { IPointer(), ci } );
		const char*	name= TF_OperationName( input.oper );
		const char*	op_name= TF_OperationOpType( input.oper );
		CC_PRINT( "  input[%d]=%s  ==> [%s] \"%s\"\n", ci, GetTypeString(type), op_name, name );
	}
}

void	CCOperation::Dump_Output()
{
	int	count= TF_OperationNumOutputs( IPointer() );
	for( int ci= 0 ; ci< count ; ci++ ){
		TF_DataType	type= TF_OperationOutputType( { IPointer(), ci } );
		int	consumer_count= TF_OperationOutputNumConsumers( { IPointer(), ci } );
		CC_PRINT( "  output[%d]=%s  consumer=%d\n", ci, GetTypeString(type), consumer_count );
	}
}

void	CCOperation::Dump()
{
	const char*	name= TF_OperationName( IPointer() );
	const char*	op_type= TF_OperationOpType( IPointer() );
	const char*	device= TF_OperationDevice( IPointer() );
	int			output_count= TF_OperationNumOutputs( IPointer() );
	int			input_count= TF_OperationNumInputs( IPointer() );
	CC_PRINT( "[%s] dev=%s name=\"%s\" input=%d output=%d\n",
				op_type,
				device,
				name,
				input_count,
				output_count
			);
	Dump_Input();
	Dump_Output();
}


//-----------------------------------------------------------------------------

CCOperationDescription::CCOperationDescription( TF_OperationDescription* ptr, CCOperation* cop ) :
	iOperation( cop )
{
	iPointer= ptr;
}

CCOperationDescription::~CCOperationDescription()
{
	Finish();
}

CCOperationDescription&	CCOperationDescription::SetAttr( const char* attr_name, CCTensor& src )
{
	CCStatus	status;
	TF_SetAttrTensor( IPointer(), attr_name, src.IPointer(), status.IPointer() );
	CCTF_CHECK_STATUS( status );
	return	*this;
}

CCOperationDescription&	CCOperationDescription::SetAttr( const char* attr_name, TF_DataType type )
{
	TF_SetAttrType( IPointer(), attr_name, type );
	return	*this;
}

CCOperationDescription&	CCOperationDescription::AddInput( const CCOperation& src )
{
	TF_AddInput( IPointer(), { src.IPointer(), 0 } );
	return	*this;
}

CCOperationDescription&	CCOperationDescription::SetDevice( const char* device )
{
	TF_SetDevice( IPointer(), device );
	return	*this;
}

bool	CCOperationDescription::Finish()
{
	if( !iOperation ){
		return	true;
	}
	assert( iOperation != nullptr );
	CCStatus	status;
	iOperation->SetPointer( TF_FinishOperation( IPointer(), status.IPointer() ) );
	CCTF_CHECK_STATUS( status );
	iOperation= nullptr;
	return	status.IsOK();
}


//-----------------------------------------------------------------------------

CCImportGraphDefOptions::CCImportGraphDefOptions()
{
	iPointer= TF_NewImportGraphDefOptions();
}

CCImportGraphDefOptions::~CCImportGraphDefOptions()
{
	Finalize();
}

void	CCImportGraphDefOptions::Finalize()
{
	if( iPointer ){
		TF_DeleteImportGraphDefOptions( iPointer );
		iPointer= nullptr;
	}
}

CCImportGraphDefOptions&	CCImportGraphDefOptions::SetDefaultDevice( const char* device )
{
	TF_ImportGraphDefOptionsSetDefaultDevice( IPointer(), device );
	return	*this;
}


//-----------------------------------------------------------------------------

CCGraph::CCGraph()
{
	Initialize();
}

CCGraph::~CCGraph()
{
	Finalize();
}

void	CCGraph::Initialize()
{
	Finalize();
	iPointer= TF_NewGraph();
}

void	CCGraph::Finalize()
{
	if( iPointer ){
		TF_DeleteGraph( iPointer );
		iPointer= nullptr;
	}
}

CCOperationDescription	CCGraph::CreateOperation( CCOperation& op, const char* op_type, const char* name )
{
	return	CCOperationDescription( TF_NewOperation( IPointer(), op_type, name ), &op );
}

bool	CCGraph::Import( const CCBuffer& buffer, const CCImportGraphDefOptions& opt )
{
	CCStatus	status;
	TF_GraphImportGraphDef( IPointer(), buffer.IPointer(), opt.IPointer(), status.IPointer() );
	CCTF_CHECK_STATUS( status );
	return	status.IsOK();
}

bool	CCGraph::Export( CCBuffer& buffer )
{
	CCStatus	status;
	TF_GraphToGraphDef( IPointer(), buffer.IPointer(), status.IPointer() );
	CCTF_CHECK_STATUS( status );
	return	status.IsOK();
}

bool	CCGraph::FindOperation( CCOperation& op, const char* name )
{
	auto*	ptr= TF_GraphOperationByName( IPointer(), name );
	if( !ptr ){
		CC_LOG( "Operation NotFound %s\n", name );
		return	false;
	}
	op.SetPointer( ptr );
	return	true;
}

void	CCGraph::GetOutputShape( CCShape& shape, const CCOperation& op, int index )
{
	CCStatus	status;
	int	rank= TF_GraphGetTensorNumDims( IPointer(), { op.IPointer(), index }, status.IPointer() );
	CCTF_CHECK_STATUS( status );
	if( rank >= 1 ){
		assert( (unsigned int)rank <= CCShape::RANK_MAX );
		int64_t	dims[CCShape::RANK_MAX];
		TF_GraphGetTensorShape( IPointer(), { op.IPointer(), index }, dims, rank, status.IPointer() );
		shape.SetDims( dims, rank );
	}else{
		CC_PRINT( "Rank=%d\n", rank );
	}
}

void	CCGraph::Dump()
{
	CCStatus	status;
	size_t	index= 0;
	for(;;){
		CCOperation	op;
		auto*	op_ptr= TF_GraphNextOperation( IPointer(), &index );
		if( !op_ptr ){
			break;
		}
		op.SetPointer( op_ptr );
		op.Dump();
		int	count= op.GetOutputCount();
		for( int ci= 0 ; ci< count ; ci++ ){
			CCShape	shape;
			GetOutputShape( shape, op, ci );
			CC_PRINT( "  out shape[%d]: ", ci );
			shape.Dump( "" );
		}
	}
}


//-----------------------------------------------------------------------------

CCSessionOptions::CCSessionOptions()
{
	iPointer= TF_NewSessionOptions();
}

CCSessionOptions::~CCSessionOptions()
{
	Finalize();
}

void	CCSessionOptions::Finalize()
{
	if( iPointer ){
		TF_DeleteSessionOptions( iPointer );
		iPointer= nullptr;
	}
}


//-----------------------------------------------------------------------------

CCSession::CCSession()
{
}

CCSession::~CCSession()
{
	Finalize();
}

void	CCSession::Finalize()
{
	if( iPointer ){
		CCStatus	status;
		TF_CloseSession( iPointer, status.IPointer() );
		CCTF_CHECK_STATUS( status );
		assert( status.IsOK() );
		TF_DeleteSession( iPointer, status.IPointer() );
		CCTF_CHECK_STATUS( status );
		assert( status.IsOK() );
		iPointer= nullptr;
	}
}

bool	CCSession::Init( CCGraph& graph, const CCSessionOptions& options )
{
	Finalize();
	CCStatus	status;
	SetPointer( TF_NewSession( graph.IPointer(), options.IPointer(), status.IPointer() ) );
	CCTF_CHECK_STATUS( status );
	return	status.IsOK();
}

bool	CCSession::Run( const CCOperation*const* op_list, int op_count, 
			const CCRunParam* in_list, int in_count,
			const CCRunParam* out_list, int out_count )
{
	constexpr int	MAX_INOUT= 8;
	CCStatus	status;
	const TF_Output*	in_ptr= nullptr;
	TF_Output			in_output[MAX_INOUT];
	TF_Tensor*const*	in_value_ptr= nullptr;
	TF_Tensor*			in_pplist[MAX_INOUT];
	if( in_count ){
		assert( in_count < MAX_INOUT );
		for( int ci= 0 ; ci< in_count ; ci++ ){
			in_output[ci]= { in_list[ci].iOp->IPointer(), in_list[ci].Index };
			in_pplist[ci]= in_list[ci].iTensor->IPointer();
		}
		in_ptr= in_output;
		in_value_ptr= in_pplist;
	}
	const TF_Output*	out_ptr= nullptr;
	TF_Output			out_output[MAX_INOUT];
	TF_Tensor**			out_value_ptr= nullptr;
	TF_Tensor*			out_pplist[MAX_INOUT];
	if( out_count ){
		assert( out_count < MAX_INOUT );
		for( int ci= 0 ; ci< out_count ; ci++ ){
			out_output[ci]= { out_list[ci].iOp->IPointer(), out_list[ci].Index };
			out_pplist[ci]= nullptr;
		}
		out_ptr= out_output;
		out_value_ptr= out_pplist;
	}
	TF_Operation**	op_ptr= nullptr;
	TF_Operation*	op_pplist[MAX_INOUT];
	if( op_count ){
		assert( op_count < MAX_INOUT );
		for( int ci= 0 ; ci< op_count ; ci++ ){
			op_pplist[ci]= op_list[ci]->IPointer();
		}
		op_ptr= op_pplist;
	}
	TF_SessionRun(
			IPointer(),
			nullptr,
			in_ptr,		in_value_ptr,	in_count,
			out_ptr,	out_value_ptr,	out_count,
			op_ptr,		op_count,
			nullptr,
			status.IPointer()
		);
	if( status.IsOK() ){
		for( int ci= 0 ; ci< out_count ; ci++ ){
			out_list[ci].iTensor->SetPointer( out_pplist[ci] );
		}
	}
	CCTF_CHECK_STATUS( status );
	return	status.IsOK();
}

bool	CCSession::Run( const TF_Operation*const* op_list, int op_count, 
			const TF_Output* in_list, TF_Tensor*const* in_value_list, int in_count,
			const TF_Output* out_list, TF_Tensor** out_value_list, int out_count )
{
	CCStatus	status;
	TF_SessionRun(
			IPointer(),
			nullptr,
			in_list,	in_value_list,	in_count,
			out_list,	out_value_list,	out_count,
			op_list,	op_count,
			nullptr,
			status.IPointer()
		);
	CCTF_CHECK_STATUS( status );
	return	status.IsOK();
}


//-----------------------------------------------------------------------------






//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
}

