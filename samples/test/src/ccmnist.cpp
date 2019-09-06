// 2019/09/04 Hiroyuki Ogasawara
// vim:ts=4 sw=4 noet:

#include	<stdio.h>
#include	<cctf/CCTFLibrary.h>
#include	<cctf/CCTFSystem.h>
#include	<random>
#include	"MNIST_Loader.h"

using namespace cctf;

void	ccmnist()
{
	constexpr int	BATCH_SIZE= 128;
	constexpr int	LOOP_COUNT= 10 * 60000/BATCH_SIZE;
	constexpr int	TEST_LOOP_COUNT= 10000/BATCH_SIZE;

	MNIST_Loader	loader;
	loader.Load();

	std::random_device	seed;
	std::mt19937	engine( seed() );

	CCGraph		graph;
	CCSession	session;
	session.Init( graph, CCSessionOptions() );

	{
		CCBuffer	buffer;
		buffer.LoadFile( "graph.pb" );
		graph.Import( buffer, CCImportGraphDefOptions() );

		{
			CCOperation	init;
			CCOperation*	op_list[]= { &init };
			graph.FindOperation( init, "init" );
			session.Run(
					op_list, 1,
					nullptr, 0,
					nullptr, 0 );
		}

		CCOperation	xinput;
		graph.FindOperation( xinput, "xinput" );

		{
			CCOperation	train;
			graph.FindOperation( train, "train" );

			CCOperation	yinput;
			graph.FindOperation( yinput, "yinput" );

			CCOperation	loss;
			graph.FindOperation( loss, "loss" );

			CCOperation*	op_list[]= { &train };

			CCTensor	x( TF_FLOAT, {BATCH_SIZE,1,28,28} );
			CCTensor	y( TF_FLOAT, {BATCH_SIZE,10} );

			CCRunParam	input[]= {
					{ xinput, 0, x },
					{ yinput, 0, y },
				};

			CCTensor	loss_val;
			CCRunParam	output[]= {
					{ loss, 0, loss_val },
				};

			std::uniform_int_distribution<unsigned int>	rand( 0, 60000-1 );
			for( int ei= 0 ; ei< LOOP_COUNT ; ei++ ){
				float*	image= x.Map<float>();
				float*	label= y.Map<float>();
				for( int bi= 0 ; bi< BATCH_SIZE ; bi++ ){
					unsigned int	ri= rand( engine );
					loader.GetImage( image, ri, false );
					loader.GetLabel10( label, ri, false );
					image+= 28*28;
					label+= 10;
				}
				session.Run(
					op_list, 1,
					input, 2,
					nullptr, 0 );
				if( ei % 100 == 0 ){
					session.Run(
						nullptr, 0,
						input, 2,
						output, 1 );
					CC_PRINT( "ei=%d/%d loss=%f\n", ei, LOOP_COUNT, loss_val.Map<float>()[0] );
					loss_val.Finalize();
				}
			}
			{
				CCTensor	path_val;
				path_val.SetString( "save/mnist" );
				CCOperation	saver;
				graph.FindOperation( saver, "save/control_dependency" );
				CCOperation	path;
				graph.FindOperation( path, "save/filename" );
				CCRunParam	input[]= {
						{	path, 0, path_val	},
					};
				CCOperation*	op_list[]= { &saver };
				session.Run(
					op_list, 1,
					input, 1,
					nullptr, 0 );
			}
		}

		{
			CCOperation	youtput;
			graph.FindOperation( youtput, "youtput" );

			CCTensor	x( TF_FLOAT, {BATCH_SIZE,1,28,28} );
			CCRunParam	input[]= {
					{ xinput, 0, x },
				};

			CCTensor	y;
			CCRunParam	output[]= {
					{ youtput, 0, y },
				};

			std::uniform_int_distribution<unsigned int>	rand( 0, 10000-1 );
			int	score= 0;
			for( int ei= 0 ; ei< TEST_LOOP_COUNT ; ei++ ){
				float*	image= x.Map<float>();
				unsigned int	labels[BATCH_SIZE];
				for( int bi= 0 ; bi< BATCH_SIZE ; bi++ ){
					unsigned int	ri= rand( engine );
					loader.GetImage( image, ri, true );
					labels[bi]= loader.GetLabel( ri, true );
					image+= 28*28;
				}
				session.Run(
					nullptr, 0,
					input, 1,
					output, 1 );
				float*	label= y.Map<float>();
				for( int bi= 0 ; bi< BATCH_SIZE ; bi++ ){
					unsigned int	result= MNIST_Loader::ArgMax10( label );
					if( result == labels[bi] ){
						score++;
					}
					label+= 10;
				}
				y.Finalize();
			}
			CC_PRINT( "score=%f\n", score*100.0f / (TEST_LOOP_COUNT * BATCH_SIZE) );
		}
	}

	CC_LOG( "OK %s\n", __func__ );
}

