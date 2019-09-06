// 2019/09/04 Hiroyuki Ogasawara
// vim:ts=4 sw=4 noet:

#include	<stdio.h>
#include	<stdlib.h>
#include	<cctf/CCTFLibrary.h>

using namespace cctf;

namespace {
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

static void	cctest_1()
{
	printf( "TensorFlow Version=%s\n", TF_Version() );
	printf( "OK %s\n", __func__ );
}

static void	cctest_2()
{
	CCShape	shape( 1, 2, 3, 4, 5 );
	assert( shape.GetX() == 5 );
	assert( shape.GetY() == 4 );
	assert( shape.GetZ() == 3 );
	assert( shape.GetW() == 2 );
	assert( shape.GetR(4) == 1 );
	assert( shape.Get(4) == 5 );
	assert( shape.GetRank() == 5 );
	assert( shape.GetElementCount() == 120 );
	shape.Set( 0, 8 );
	shape.SetR( 1, 9 );
	assert( shape.GetR(4) == 8 );
	assert( shape.Get(3) == 9 );

	printf( "OK %s\n", __func__ );
}

static void	cctest_3()
{
	CCGraph		graph;
	CCSession	session;
	session.Init( graph, CCSessionOptions() );
	{
		static float		init_data1[]= {
			1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f,
		};
		static float		init_data2[]= {
			1.0f, 1.0f, 0.5f, 0.5f, 0.1f, 0.1f, 0.0f, -1.0f, -2.0f, -3.0f,
		};

		CCTensor	tensor1;
		tensor1.Set( {10}, init_data1, 10 );

		CCTensor	tensor2;
		tensor2.Set( {10}, init_data2, 10 );

		tensor1.Dump();
		tensor2.Dump();


		CCOperation	parama;
		graph.CreateOperation( parama, "Const", "a" )
			.SetAttr( "value", tensor1 )
			.SetAttr( "dtype", TF_FLOAT )
			.Finish();

		CCOperation	paramb;
		graph.CreateOperation( paramb, "Const", "b" )
			.SetAttr( "value", tensor2 )
			.SetAttr( "dtype", TF_FLOAT )
			.Finish();

		CCOperation	op_add;
		graph.CreateOperation( op_add, "Add", "add" )
			.AddInput( parama )
			.AddInput( paramb )
			.Finish();

		graph.Dump();

		{
			CCTensor	result;
			CCRunParam	output{ op_add, 0, result };
			session.Run(
					nullptr, 0,
					nullptr, 0,
					&output, 1 );
			result.Dump( "Result" );

			const float*	fptr= result.Map<float>();
			for( int i= 0 ; i< 10 ; i++ ){
				assert( fptr[i] == init_data1[i] + init_data2[i] );
			}
		}
	}
	printf( "OK %s\n", __func__ );
}


//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
}

void	cctest()
{
	cctest_1();
	cctest_2();
	cctest_3();
}




