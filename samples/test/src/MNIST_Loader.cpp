// twrapper 2018/11/16 Hiroyuki Ogasawara
// vim:ts=4 sw=4 noet:

#include	<cctf/CCTFSystem.h>
#include	<string.h>
#include	<assert.h>
#include	<stdint.h>
#include	<stdio.h>
#include	"MNIST_Loader.h"

using namespace cctf;


MNIST_Loader::MNIST_Loader()
{
	memset( iImage, 0, sizeof(iImage) );
}

MNIST_Loader::~MNIST_Loader()
{
	Finalize();
}

void	MNIST_Loader::Finalize()
{
	for( unsigned int fi= 0 ; fi< FILE_COUNT ; fi++ ){
		CCSystem::ZFree( iImage[fi], 0 );
	}
}

void	MNIST_Loader::Load()
{
	static const char*	file_list[]= {
		"data/train-images.idx3-ubyte",
		"data/t10k-images.idx3-ubyte",
		"data/train-labels.idx1-ubyte",
		"data/t10k-labels.idx1-ubyte",
	};
	for( unsigned int fi= 0 ; fi< FILE_COUNT ; fi++ ){
		size_t	size= 0;
		iImage[fi]= CCSystem::LoadFile( file_list[fi], size );
	}
}

void	MNIST_Loader::GetImage( float* str, unsigned int index, bool test )
{
	assert( index < 60000 );
	const uint8_t*	data_ptr= reinterpret_cast<uint8_t*>( iImage[(int)test] ) + HEADER_OFFSET_IMAGE;
	const uint8_t*	ptr= data_ptr + IMAGE_BYTE_SIZE * index;
	constexpr float	FloatScale= 1.0f/255.0f;
	for( unsigned int bi= 0 ; bi< IMAGE_BYTE_SIZE ; bi++ ){
		*str++= *ptr++ * FloatScale;
	}
}

unsigned int	MNIST_Loader::GetLabel( unsigned int index, bool test ) const
{
	assert( index < 60000 );
	auto	label= reinterpret_cast<uint8_t*>( iImage[2+(int)test] )[HEADER_OFFSET_LABEL + index];
	assert( label < LABEL_COUNT );
	return	label;
}

void	MNIST_Loader::GetLabel10( float* buffer, unsigned int index, bool test )
{
	auto	label= GetLabel( index, test );
	memset( buffer, 0, sizeof(float) * LABEL_COUNT );
	buffer[label]= 1.0f;
}

unsigned int	MNIST_Loader::ArgMax10( const float* buffer )
{
	float	max_value= -1e20f;
	unsigned int	index= 0;
	for( unsigned int ai= 0 ; ai < LABEL_COUNT ; ai++ ){
		if( buffer[ai] > max_value ){
			max_value= buffer[ai];
			index= ai;
		}
	}
	return	index;
}



