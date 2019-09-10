#include "mfem.hpp"
#include <fstream>
#include <iostream>

#include "multigrid.hpp"
#include "eigenvalue.hpp"

using namespace std;
using namespace mfem;

void getEssentialTrueDoFs(Mesh* mesh, FiniteElementSpace* fespace, Array<int>& ess_tdof_list)
{
   if (mesh->bdr_attributes.Size())
   {
      Array<int> ess_bdr(mesh->bdr_attributes.Max());
      ess_bdr = 1;
      fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
   }
}

int main(int argc, char *argv[])
{
   int num_procs, myid;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   // 1. Parse command-line options.
   // const char *mesh_file = "../../data/fichera.mesh";
   int ref_levels = 2;
   int mg_levels = 2;
   int order = 3;
   const char *basis_type = "G"; // Gauss-Lobatto
   bool visualization = 1;

   OptionsParser args(argc, argv);
   // args.AddOption(&mesh_file, "-m", "--mesh",
   //                "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Order of the finite element spaces");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the initial mesh uniformly;"
                  "This mesh will be the coarse mesh in the multigrid hierarchy");
   args.AddOption(&mg_levels, "-l", "--levels",
                  "Number of levels in the multigrid hierarchy;");
   args.AddOption(&basis_type, "-b", "--basis-type",
                  "Basis: G - Gauss-Lobatto, P - Positive, U - Uniform");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();
   if (!args.Good())
   {
      if (myid == 0)
      {
         args.PrintUsage(cout);
      }
      MPI_Finalize();
      return 1;
   }

   if (myid == 0)
   {
      args.PrintOptions(cout);
   }

   // See class BasisType in fem/fe_coll.hpp for available basis types
   int basis = BasisType::GetType(basis_type[0]);
   if (myid == 0)
   {
      cout << "Using " << BasisType::Name(basis) << " basis ..." << endl;
   }

   // 2. Read the mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral, hexahedral, surface and volume meshes with
   //    the same code.
   // Mesh *mesh = new Mesh(mesh_file, 1, 1);
   // Mesh *mesh = new Mesh(1, 1, 1, Element::HEXAHEDRON, true, 1.0, 1.0, 1.0, false);
   Mesh *mesh = new Mesh(1, 1, Element::QUADRILATERAL, true, 1.0, 1.0, false);
   int dim = mesh->Dimension();

   // Initial refinements of the input grid
   for (int i = 0; i < ref_levels; ++i)
   {
      mesh->UniformRefinement();
   }

   ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
   delete mesh;
   mesh = nullptr;

   FiniteElementCollection *fec = new H1_FECollection(order, dim, basis);

   // Set up coarse grid finite element space
   ParFiniteElementSpace* fespace = new ParFiniteElementSpace(pmesh, fec);
   HYPRE_Int size = fespace->GlobalTrueVSize();

   if (myid == 0)
   {
      cout << "Number of finite element unknowns on level 0: " << size << endl;
   }

   Array<int>* essentialTrueDoFs = new Array<int>();
   getEssentialTrueDoFs(pmesh, fespace, *essentialTrueDoFs);

   // Construct hierarchy of finite element spaces
   ParSpaceHierarchy spaceHierarchy(pmesh, fespace);

   for (int level = 1; level < mg_levels; ++level)
   {
      ParMesh* nextMesh = new ParMesh(*spaceHierarchy.GetMesh(level - 1));
      nextMesh->UniformRefinement();
      fespace = new ParFiniteElementSpace(nextMesh, fec);
      spaceHierarchy.addLevel(nextMesh, fespace);

      size = fespace->GlobalTrueVSize();

      if (myid == 0)
      {
         cout << "Number of finite element unknowns on level " << level << ": " << size << endl;
      }
   }

   ParBilinearForm* coarseForm = new ParBilinearForm(spaceHierarchy.GetFESpace(0));
   coarseForm->SetAssemblyLevel(AssemblyLevel::PARTIAL);
   ConstantCoefficient one(1.0);
   coarseForm->AddDomainIntegrator(new DiffusionIntegrator(one));
   coarseForm->Assemble();
   OperatorPtr coarseOpr;
   coarseForm->FormSystemMatrix(*essentialTrueDoFs, coarseOpr);

   CGSolver* coarseSolver = new CGSolver(MPI_COMM_WORLD);
   coarseSolver->SetPrintLevel(-1);
   coarseSolver->SetMaxIter(100);
   coarseSolver->SetRelTol(1e-4);
   coarseSolver->SetAbsTol(0.0);
   coarseSolver->SetOperator(*coarseOpr);

   OperatorMultigrid oprMultigrid(coarseForm, coarseOpr.Ptr(), coarseSolver, essentialTrueDoFs);

   for(int level = 1; level < spaceHierarchy.GetNumLevels(); ++level)
   {
      // Operator
      tic_toc.Clear();
      tic_toc.Start();
      if (myid == 0)
      {
         cout << "Partially assemble HPC form on level " << level << "..." << flush;
      }

      ParBilinearForm* form = new ParBilinearForm(spaceHierarchy.GetFESpace(level));
      form->SetAssemblyLevel(AssemblyLevel::PARTIAL);
      form->AddDomainIntegrator(new DiffusionIntegrator(one));
      form->Assemble();
      OperatorPtr* opr = new OperatorPtr();
      essentialTrueDoFs = new Array<int>();
      getEssentialTrueDoFs(spaceHierarchy.GetMesh(level), spaceHierarchy.GetFESpace(level), *essentialTrueDoFs);
      form->FormSystemMatrix(*essentialTrueDoFs, *opr);
      tic_toc.Stop();
      if (myid == 0)
      {
         cout << "\tdone, " << tic_toc.RealTime() << "s." << endl;
      }

      // Smoother
      tic_toc.Clear();
      tic_toc.Start();
      if (myid == 0)
      {
         cout << "Partially assemble diagonal on level " << level << "..." << flush;
      }
      Vector diag(spaceHierarchy.GetFESpace(level)->GetTrueVSize());
      form->AssembleDiagonal(diag);
      OperatorJacobiSmoother* pa_smoother_one = new OperatorJacobiSmoother(diag, *essentialTrueDoFs, 1.0);
      tic_toc.Stop();
      if (myid == 0)
      {
         cout << "\tdone, " << tic_toc.RealTime() << "s." << endl;
      }

      // Prolongation
      tic_toc.Clear();
      tic_toc.Start();
      if (myid == 0)
      {
         cout << "Create matrix-free P/R operator on " << level << "..." << flush;
      }
      OperatorHandle* P = new OperatorHandle(Operator::ANY_TYPE);
      spaceHierarchy.GetFESpace(level)->GetTrueTransferOperator(*spaceHierarchy.GetFESpace(level - 1), *P);
      Operator* prolongation = P->Ptr();
      tic_toc.Stop();
      if (myid == 0)
      {
         cout << "\t\tdone, " << tic_toc.RealTime() << "s." << endl;
      }

      tic_toc.Clear();
      tic_toc.Start();
      if (myid == 0)
      {
         cout << "Estimating eigenvalues on level " << level << "..." << flush;
      }
      Vector ev(spaceHierarchy.GetFESpace(level)->GetTrueVSize());
      ProductOperator DinvA(pa_smoother_one, opr->Ptr(), false, false);
      double eigval = PowerMethod::EstimateLargestEigenvalue(DinvA, ev, 20, 1e-8);
      tic_toc.Stop();
      if (myid == 0)
      {
         cout << "\t\tdone, " << tic_toc.RealTime() << "s." << endl;
      }

      OperatorChebyshevSmoother* pa_smoother = new OperatorChebyshevSmoother(opr->Ptr(), diag, *essentialTrueDoFs, 5, eigval);

      oprMultigrid.AddLevel(form, opr->Ptr(), pa_smoother, prolongation, essentialTrueDoFs);
   }

   ParGridFunction x(spaceHierarchy.GetFinestFESpace());
   x = 0.0;

   if (myid == 0)
   {
      cout << "Assembling rhs..." << flush;
   }
   tic_toc.Clear();
   tic_toc.Start();
   ParLinearForm b(spaceHierarchy.GetFinestFESpace());
   b.AddDomainIntegrator(new DomainLFIntegrator(one));
   b.Assemble();
   tic_toc.Stop();
   if (myid == 0)
   {
      cout << "\t\t\t\tdone, " << tic_toc.RealTime() << "s." << endl;
   }

   Vector X, B;
   OperatorPtr dummy;

   oprMultigrid.GetFormAtFinestLevel()->FormLinearSystem(*oprMultigrid.GetEssentialDoFsAtFinestLevel(), x, b, dummy, X, B);

   Vector r(X.Size());
   SolverMultigrid vCycle(oprMultigrid);

   // CGSolver pcg(MPI_COMM_WORLD);
   // pcg.SetPrintLevel(1);
   // pcg.SetMaxIter(10);
   // pcg.SetRelTol(1e-5);
   // pcg.SetAbsTol(0.0);
   // pcg.SetOperator(*oprMultigrid.GetOperatorAtFinestLevel());
   // pcg.SetPreconditioner(vCycle);
   // pcg.Mult(B, X);

   oprMultigrid.Mult(X, r);
   subtract(B, r, r);

   double beginRes = r * r;
   double prevRes = beginRes;
   const int printWidth = 11;

   if (myid == 0)
   {
      cout << std::setw(3) << "It";
      cout << std::setw(printWidth) << "Absres";
      cout << std::setw(printWidth) << "Relres";
      cout << std::setw(printWidth) << "Conv";
      cout << std::setw(printWidth) << "Time [s]" << endl;

      cout << std::setw(3) << 0;
      cout << std::scientific << std::setprecision(3) << std::setw(printWidth) << beginRes;
      cout << std::scientific << std::setprecision(3) << std::setw(printWidth) << 1.0;
      cout << std::scientific << std::setprecision(3) << std::setw(printWidth) << 0.0;
      cout << std::scientific << std::setprecision(3) << std::setw(printWidth) << 0.0 << endl;
   }

   for (int iter = 0; iter < 10; ++iter)
   {
      tic_toc.Clear();
      tic_toc.Start();
      vCycle.Mult(B, X);
      tic_toc.Stop();

      oprMultigrid.Mult(X, r);
      subtract(B, r, r);

      double res = r * r;
      if (myid == 0)
      {
         cout << std::setw(3) << iter + 1;
         cout << std::scientific << std::setprecision(3) << std::setw(printWidth) << res;
         cout << std::scientific << std::setprecision(3) << std::setw(printWidth) << res/beginRes;
         cout << std::scientific << std::setprecision(3) << std::setw(printWidth) << res/prevRes;
         cout << std::scientific << std::setprecision(3) << std::setw(printWidth) << tic_toc.RealTime() << endl;
      }

      if (res < 1e-10 * beginRes)
      {
         break;
      }

      prevRes = res;
   }

   oprMultigrid.GetFormAtFinestLevel()->RecoverFEMSolution(X, b, x);

   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock << "parallel " << num_procs << " " << myid << "\n";
      sol_sock.precision(8);
      sol_sock << "solution\n" << *spaceHierarchy.GetFinestMesh() << x << flush;
   }

   // Missing a bunch of deletes
   
   MPI_Finalize();

   return 0;
}