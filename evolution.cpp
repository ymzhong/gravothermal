////////////////////////////////////////////////////////////////////////////////////////////////
// Use Eigen library as the linear equation solver                                            //
// add static plummer baryons to the NFW dark matter halo
////////////////////////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <fstream>
#include <ctime>
#include <cmath>
#include <string>
#include <iomanip>
#include <vector>
#include <stdexcept>
#include <sstream>
#include <memory>
#include <Eigen/Dense>

// Forward declarations
class SimulationParameters;
class SimulationState;
class FileManager;
class Logger;

// Simulation configuration
// total step to run the simulation
constexpr int DEFAULT_TOTAL_STEPS = 1000000000;
// save step to save the simulation state
constexpr int DEFAULT_SAVE_STEPS = 100;
// default epsilon for the simulation
// this is the maximum absolute value change in internal energy
constexpr double DEFAULT_EPSILON = 0.001;
// default iteration steps for relaxation
constexpr int DEFAULT_RELAXATION_STEPS = 10;
// default density threshold for stopping the simulation
constexpr double DEFAULT_DENSITY_THRESHOLD = 1e30;


/**
 * Logger class to handle output messages
 */
class Logger {
private:
    bool verbose;
    
public:
    explicit Logger(bool isVerbose = false) : verbose(isVerbose) {}
    
    void setVerbose(bool isVerbose) {
        verbose = isVerbose;
    }
    
    void info(const std::string& message) const {
        std::cout << message << std::endl;
    }
    
    void debug(const std::string& message) const {
        if (verbose) {
            std::cout << "[DEBUG] " << message << std::endl;
        }
    }
    
    void error(const std::string& message) const {
        std::cerr << "[ERROR] " << message << std::endl;
    }
};

/**
 * Class to handle simulation parameters
 */
class SimulationParameters {
public:
    // Simulation control parameters
    int totalStep;
    int saveStep;
    double epsilon;
    double totalTime;
    
    // Cross section parameters
    double a;
    double b;
    double c;
    double sigma;
    
    // Baryon enclosed mass function parameters
    double mass_norm;
    double scale_norm;
    
    // IO parameters
    std::string tag;
    std::string inputDir;
    std::string outputFile;
    
    // Model parameters
    SimulationParameters() : 
        totalStep(DEFAULT_TOTAL_STEPS),
        saveStep(DEFAULT_SAVE_STEPS),
        epsilon(DEFAULT_EPSILON),
        // initial time
        totalTime(0.0),
        // a,b,c are the parameters for the SIDM conductivity terms
        a(2.257),
        b(1.385),
        c(0.753),
        // sigma is the normalized SIDM cross section (sigma/m)*rho_s*r_s
        sigma(0.5),
        // mass_norm is the normalized baryon mass, M_b/(4*pi*rho_s*r_s^3)
        mass_norm(0.1),
        // scale_norm is the normalized baryon scale radius, a/r_s
        scale_norm(0.1),
        // tag to identify the inital profiles and name the output file
        // this should be the same as the one used in the input files
        tag("20250402A"),
        inputDir(""),
        outputFile("") {
        
        // Set default directories for input and output files
        // Note: Adjust these paths as per your directory structure
        inputDir = "/path/to/initial/" + tag + "/";
        outputFile = "/path/to/output/result_" + tag + ".txt";
    }
    
    // Display parameters to console
    void display(const Logger& logger) const {
        logger.info("Initial profile: " + tag);
        logger.info("Initial time: " + std::to_string(totalTime));
        logger.info("Total step: " + std::to_string(totalStep));
        logger.info("Save step: " + std::to_string(saveStep));
        logger.info("Abs(delta u/u): " + std::to_string(epsilon));
        logger.info("Cross section (sigma): " + std::to_string(sigma));
        logger.info("Conduction parameter (a, b, c): " + std::to_string(a) + ", " 
                 + std::to_string(b) + ", " + std::to_string(c));
        logger.info("Baryon parameter (mass_norm, scale_norm): " + std::to_string(mass_norm) + ", "
                 + std::to_string(scale_norm));
    }
};

// Baryon enclosed mass function implementations
// Plummer model, the default one
double MbaryonP(double myr, double mymass, double myscale) {
    // Optimized: replaced std::pow(myscale, 2.0) and std::pow(myr, 2.0)
    return mymass * std::pow(1.0 + (myscale * myscale)/(myr * myr), -1.5);
}
// Hernquist model
double MbaryonH(double myr, double mymass, double myscale) {
    return mymass * std::pow(1.0 + myscale/myr, -2.0);
}
// Single-Power-Law model
double MbaryonSPL(double myr, double mymass) {
    return mymass * std::pow(myr, 0.6);
}

/**
 * Class to handle file operations
 */
class FileManager {
private:
    Logger& logger;

public:
    explicit FileManager(Logger& log) : logger(log) {}
    
    // Read matrix from file with error handling
    Eigen::MatrixXd readMatrix(const std::string& filename) const {
        std::vector<double> buffer;
        int cols = 0, rows = 0;
        
        std::ifstream infile(filename);
        if (!infile) {
            throw std::runtime_error("Cannot open file: " + filename);
        }
        
        std::string line;
        while (std::getline(infile, line)) {
            std::istringstream stream(line);
            
            int temp_cols = 0;
            double value;
            while (stream >> value) {
                buffer.push_back(value);
                temp_cols++;
            }
            
            if (temp_cols == 0) {
                continue;
            }
            
            if (cols == 0) {
                cols = temp_cols;
            }
            
            rows++;
        }
        
        infile.close();
        
        // Populate matrix with numbers
        Eigen::MatrixXd result(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result(i, j) = buffer[cols * i + j];
            }
        }
        
        return result;
    }
    
    // Write simulation parameters to file
    void writeParameters(const SimulationParameters& params, 
                         const SimulationState& state) const;
    
    // Write simulation state to file
    void writeState(const std::string& filename, 
                    double totalTime, 
                    int step, 
                    const SimulationState& state) const;
};

/**
 * Class to manage the simulation state
 */
class SimulationState {
public:
    Eigen::ArrayXd RList;      // Radius
    Eigen::ArrayXd RhoList;    // Density
    Eigen::ArrayXd MList;      // Mass (dark matter only)
    Eigen::ArrayXd MhyList;    // Total mass (dark matter + baryon)
    Eigen::ArrayXd uList;      // Specific internal energy
    Eigen::ArrayXd LList;      // Luminosity
    Eigen::ArrayXd vList;      // 1D Velocity dispersion
    Eigen::ArrayXd pList;      // Pressure
    Eigen::ArrayXd aList;      // Adibatic variable
    int NoLayers;              // Number of layers
    
    SimulationState() : NoLayers(0) {}
    
    // Initialize state from files
    void initialize(const SimulationParameters& params, 
                   const FileManager& fileManager, 
                   Logger& logger) {
        try {
            // Construct input file paths
            std::string nameR = params.inputDir + "RList-" + params.tag + ".txt";
            std::string nameRho = params.inputDir + "RhoList-" + params.tag + ".txt";
            std::string nameM = params.inputDir + "MList-" + params.tag + ".txt";
            std::string nameu = params.inputDir + "uList-" + params.tag + ".txt";
            std::string nameL = params.inputDir + "LList-" + params.tag + ".txt";
            
            // Read data
            RList = fileManager.readMatrix(nameR).array();
            RhoList = fileManager.readMatrix(nameRho).array();
            MList = fileManager.readMatrix(nameM).array();
            uList = fileManager.readMatrix(nameu).array();
            LList = fileManager.readMatrix(nameL).array();
            
            NoLayers = RList.rows();
            
            // Initialize arrays
            MhyList = Eigen::ArrayXd::Zero(NoLayers);
            vList = Eigen::ArrayXd::Zero(NoLayers);
            pList = Eigen::ArrayXd::Zero(NoLayers);
            aList = Eigen::ArrayXd::Zero(NoLayers);
            
            // Set initial 1D velocity dispersion
            vList = std::sqrt(2.0/3.0) * uList.sqrt();
            
            // Add baryon mass, default is Plummer model
            for (int i = 0; i < (NoLayers-1); i++) {
                MhyList(i) = MList(i) + MbaryonP(RList(i), params.mass_norm, params.scale_norm);
            }
            
            // Log initialization
            logger.info("Number of layers: " + std::to_string(NoLayers));
            logger.info("Initial r inner most: " + std::to_string(RList(0)));
            logger.info("Initial r next to outer most: " + std::to_string(RList(NoLayers-1)));
            logger.info("Initial rho inner most: " + std::to_string(RhoList(0)));
            logger.info("Initial rho next to outer most: " + std::to_string(RhoList(NoLayers-1)));
            
        } catch (const std::exception& e) {
            logger.error("Failed to initialize simulation state: " + std::string(e.what()));
            throw;
        }
    }
    
    // Check for abnormal state that would require stopping the simulation
    bool checkForAbnormalState(Logger& logger) const {
        if (RhoList(0) > DEFAULT_DENSITY_THRESHOLD) {
            logger.info("Rho reaches threshold!");
            return true;
        }
        
        if (RList(0) < 0) {
            logger.info("R(0) is negative!");
            return true;
        }
        
        if (std::isnan(RList(0))) {
            logger.info("R is nan!");
            return true;
        }
        
        return false;
    }
};



/**
 * Class handling the simulation evolution
 */
class Simulator {
private:
    SimulationParameters params;
    SimulationState state;
    FileManager fileManager;
    Logger logger;
    
    // Helper arrays for evolution calculations
    Eigen::ArrayXd deltaUcoeff;
    Eigen::ArrayXd deltaU;
    Eigen::ArrayXd hydrostaticI;
    Eigen::ArrayXd hydrostaticM;
    Eigen::ArrayXd hydrostaticF;
    Eigen::MatrixXd Hydromat;
    Eigen::VectorXd Hydrob;
    Eigen::VectorXd deltaR;
    Eigen::VectorXd deltap;
    Eigen::VectorXd deltaRho;

public:
    Simulator(const SimulationParameters& p, 
              const FileManager& fm, 
              Logger& log) : 
        params(p), 
        fileManager(fm), 
        logger(log) {
    }
    
    void initialize() {
        state.initialize(params, fileManager, logger);
        
        // Initialize helper arrays based on the number of layers
        int NoLayers = state.NoLayers;
        
        deltaUcoeff = Eigen::ArrayXd::Zero(NoLayers);
        deltaU = Eigen::ArrayXd::Zero(NoLayers);
        hydrostaticI = Eigen::ArrayXd::Zero(NoLayers);
        hydrostaticM = Eigen::ArrayXd::Zero(NoLayers);
        hydrostaticF = Eigen::ArrayXd::Zero(NoLayers);
        
        Hydromat = Eigen::MatrixXd::Zero((NoLayers-1), (NoLayers-1));
        Hydrob = Eigen::VectorXd::Zero(NoLayers-1);
        deltaR = Eigen::VectorXd::Zero(NoLayers-1);
        deltap = Eigen::VectorXd::Zero(NoLayers-1);
        deltaRho = Eigen::VectorXd::Zero(NoLayers-1);
        
        // Write initial parameters to output file
        std::ofstream file(params.outputFile, std::ofstream::out | std::ofstream::app);
        if (file.is_open()) {
            file << "Input profile name: " << params.tag << '\n'
                << "Initial time: "  << params.totalTime << '\n'
                << "Number of layers: " << state.NoLayers << '\n'
                << "Initial r inner most: " << state.RList(0) << '\n'
                << "Initial r outer most: " << state.RList(NoLayers-1) << '\n'
                << "Initial rho inner most: " << state.RhoList(0) << '\n'
                << "Initial rho outer most: " << state.RhoList(NoLayers-1) << '\n'
                << "Total step: " << params.totalStep << '\n'
                << "Save step: " << params.saveStep << '\n'
                << "Abs(delta u/u): " << params.epsilon << '\n'
                << "Cross section (sigma): " << params.sigma << '\n'
                << "Conduction parameter (a, b, c): " << params.a << ", " << params.b << ", " << params.c << '\n'
                << "Baryon parameter (massnorm, scalenorm): " << params.mass_norm << ", " << params.scale_norm << '\n'
                << "time, step, SIDM radius, SIDM density, SIDM enclosed mass, SIDM internal energy, SIDM luminosity" << '\n'
                //output initial profiles
                << std::scientific << std::setprecision(10) << params.totalTime << " " << 0 << '\n'
                << state.RList.transpose() << '\n'
                << state.RhoList.transpose() << '\n'
                << state.MList.transpose() << '\n'
                << state.uList.transpose() << '\n'
                << state.LList.transpose() << '\n';
        }
        file.close();
        
        logger.info("Evolution with baryon starts!");
    }
    
    void runSimulation() {
        std::ofstream outputFile(params.outputFile, std::ofstream::out | std::ofstream::app);
        if (!outputFile.is_open()) {
            throw std::runtime_error("Failed to open output file: " + params.outputFile);
        }
        
        // Initialize currentSaveStep with the default value
        int currentSaveStep = params.saveStep;
        
        for (int tstep = 1; tstep < (params.totalStep + 1); tstep++) {
            try {
                // Conduction period
                double deltat = performConductionStep();
                params.totalTime += deltat;
                
                // Relaxation period
                performRelaxationStep();
                
                // Update profiles after relaxation
                updateProfiles();
                
                // Check if density reaches 1e6 threshold, adjust save frequency to every step
                if (state.RhoList(0) >= 1e6 && currentSaveStep != 1) {
                    currentSaveStep = 1;
                    logger.info("Density threshold 1e6 reached! Save frequency increased to every step.");
                }
                
                // Check for abnormal state
                if (state.checkForAbnormalState(logger)) {
                    break;
                }
                
                // Save results based on the current save step frequency
                if ((tstep % currentSaveStep) == 0) {
                    saveResults(outputFile, tstep);
                    logger.debug("Saved at time = " + std::to_string(params.totalTime) + 
                               ", step = " + std::to_string(tstep));
                }
            } catch (const std::exception& e) {
                logger.error("Error during simulation step " + std::to_string(tstep) + 
                          ": " + std::string(e.what()));
                break;
            }
        }
        
        outputFile.close();
        logger.info("Evolution ends");
    }
    
private:
    double performConductionStep() {
        int NoLayers = state.NoLayers;
        
        // Determine the time step
        deltaUcoeff(0) = -(state.LList(0) / state.MList(0)) / state.uList(0);
        for (int i = 1; i < (NoLayers-1); i++) {
            deltaUcoeff(i) = -((state.LList(i) - state.LList(i-1)) / 
                              (state.MList(i) - state.MList(i-1))) / state.uList(i);
        }
        
        double deltat = params.epsilon / (deltaUcoeff.abs().maxCoeff());
        
        // Update internal energy due to conduction
        deltaU = deltaUcoeff * state.uList * deltat;
        state.uList += deltaU;
        
        // Update pressure and adibatic variable
        state.pList = (2.0/3.0) * (state.RhoList * state.uList);
        state.aList = (2.0/3.0) * (state.RhoList.pow(-2.0/3.0) * state.uList);
        
        return deltat;
    }
    
    void performRelaxationStep() {
        int NoLayers = state.NoLayers;
        
        // Relaxation iterations
        for (int hstep = 0; hstep < DEFAULT_RELAXATION_STEPS; hstep++) {
            try {
                // Set up hydrostatic matrix
                setupHydrostaticMatrix();
                
                // Solve the system using LLT alorithm
                // Note: Eigen::LLT is used for symmetric positive definite matrices
                deltaR = Hydromat.llt().solve(Hydrob);
                
                // Calculate delta_Rho and delta_p due to the R change
                calculateDeltaRhoAndP();
                
                // Update R, p, Rho profiles
                for (int i = 0; i < (NoLayers-1); i++) {
                    state.RhoList(i) += deltaRho(i);
                    state.pList(i) += deltap(i);
                    state.RList(i) += deltaR(i);
                }
            } catch (const std::exception& e) {
                logger.error("Error during relaxation step: " + std::string(e.what()));
                throw;
            }
        }
    }
    
    void setupHydrostaticMatrix() {
        int NoLayers = state.NoLayers;
        
        // Setup first row
        double R0 = state.RList(0);
        double R0_2 = R0 * R0;
        double R0_3 = R0_2 * R0;
        double R0_4 = R0_2 * R0_2;
        double R1 = state.RList(1);
        double R1_2 = R1 * R1;
        double R1_3 = R1_2 * R1;
        double R1_R0 = R1 - R0;
        double R1_R0_sum_of_squares = R1_2 + R1*R0 + R0_2;
        double R1_3_minus_R0_3_inv = 1.0 / (R1_R0 * R1_R0_sum_of_squares);
        
        Hydromat(0, 0) = 8.0 * R0 * (state.pList(1) - state.pList(0)) + 
                        20.0 * R0_4 * 
                        (state.pList(0) / R0_3 + 
                        state.pList(1) * R1_3_minus_R0_3_inv) + 
                        3.0 * state.MhyList(0) * R1 * R0_2 * 
                        (-state.RhoList(0) / R0_3 + 
                        state.RhoList(1) * R1_3_minus_R0_3_inv);
        
        Hydromat(0, 1) = state.MhyList(0) * (state.RhoList(0) + state.RhoList(1)) - 
                        20.0 * R0_2 * state.pList(1) * 
                        R1_2 * R1_3_minus_R0_3_inv - 
                        3.0 * state.MhyList(0) * state.RhoList(1) * R1_3 * 
                        R1_3_minus_R0_3_inv;
        
        Hydrob(0) = -4.0 * R0_2 * (state.pList(1) - state.pList(0)) - 
                    state.MhyList(0) * (state.RhoList(0) + state.RhoList(1)) * R1;
        
        // Setup middle rows
        for (int i = 1; i < (NoLayers-2); ++i) {
            double Ri = state.RList(i);
            double Ri_2 = Ri * Ri;
            double Ri_3 = Ri_2 * Ri;
            double Ri_4 = Ri_2 * Ri_2;
            double Ri_1 = state.RList(i-1);
            double Ri_1_2 = Ri_1 * Ri_1;
            double Ri_1_3 = Ri_1_2 * Ri_1;
            double Ri1 = state.RList(i+1);
            double Ri1_2 = Ri1 * Ri1;
            double Ri1_3 = Ri1_2 * Ri1;
            double Ri_Ri_1 = Ri - Ri_1;
            double Ri_Ri_1_sum_of_squares = Ri_2 + Ri*Ri_1 + Ri_1_2;
            double Ri_3_minus_Ri_1_3_inv = 1.0 / (Ri_Ri_1 * Ri_Ri_1_sum_of_squares);
            double Ri1_Ri = Ri1 - Ri;
            double Ri1_Ri_sum_of_squares = Ri1_2 + Ri1*Ri + Ri_2;
            double Ri1_3_minus_Ri_3_inv = 1.0 / (Ri1_Ri * Ri1_Ri_sum_of_squares);
            
            Hydromat(i, i-1) = -state.MhyList(i) * (state.RhoList(i) + state.RhoList(i+1)) - 
                              20.0 * Ri_2 * state.pList(i) * 
                              Ri_1_2 * Ri_3_minus_Ri_1_3_inv + 
                              3.0 * state.MhyList(i) * (Ri1 - Ri_1) * 
                              state.RhoList(i) * Ri_1_2 * 
                              Ri_3_minus_Ri_1_3_inv;
            
            Hydromat(i, i) = 8.0 * Ri * (state.pList(i+1) - state.pList(i)) + 
                            20.0 * Ri_4 * 
                            (state.pList(i) * Ri_3_minus_Ri_1_3_inv + 
                            state.pList(i+1) * Ri1_3_minus_Ri_3_inv) + 
                            3.0 * state.MhyList(i) * (Ri1 - Ri_1) * 
                            Ri_2 * 
                            (-state.RhoList(i) * Ri_3_minus_Ri_1_3_inv + 
                            state.RhoList(i+1) * Ri1_3_minus_Ri_3_inv);
            
            Hydromat(i, i+1) = state.MhyList(i) * (state.RhoList(i) + state.RhoList(i+1)) - 
                              20.0 * Ri_2 * state.pList(i+1) * 
                              Ri1_2 * Ri1_3_minus_Ri_3_inv - 
                              3.0 * state.MhyList(i) * (Ri1 - Ri_1) * 
                              state.RhoList(i+1) * Ri1_2 * 
                              Ri1_3_minus_Ri_3_inv;
            
            Hydrob(i) = -4.0 * Ri_2 * (state.pList(i+1) - state.pList(i)) - 
                        state.MhyList(i) * (state.RhoList(i) + state.RhoList(i+1)) * 
                        (Ri1 - Ri_1);
        }
        
        // Setup last row
        int last = NoLayers - 2;
        double Rl = state.RList(last);
        double Rl_2 = Rl * Rl;
        double Rl_3 = Rl_2 * Rl;
        double Rl_4 = Rl_2 * Rl_2;
        double Rl_1 = state.RList(last-1);
        double Rl_1_2 = Rl_1 * Rl_1;
        double Rl_1_3 = Rl_1_2 * Rl_1;
        double Rl1 = state.RList(last+1);
        double Rl_Rl_1 = Rl - Rl_1;
        double Rl_Rl_1_sum_of_squares = Rl_2 + Rl*Rl_1 + Rl_1_2;
        double Rl_3_minus_Rl_1_3_inv = 1.0 / (Rl_Rl_1 * Rl_Rl_1_sum_of_squares);
        
        Hydromat(last, last-1) = -state.MhyList(last) * 
                               (state.RhoList(last) + state.RhoList(last+1)) - 
                               20.0 * Rl_2 * 
                               state.pList(last) * Rl_1_2 * 
                               Rl_3_minus_Rl_1_3_inv + 
                               3.0 * state.MhyList(last) * 
                               (Rl1 - Rl_1) * 
                               state.RhoList(last) * Rl_1_2 * 
                               Rl_3_minus_Rl_1_3_inv;
        
        Hydromat(last, last) = 8.0 * Rl * 
                             (state.pList(last+1) - state.pList(last)) + 
                             20.0 * Rl_4 * 
                             state.pList(last) * Rl_3_minus_Rl_1_3_inv - 
                             3.0 * state.MhyList(last) * 
                             (Rl1 - Rl_1) * 
                             Rl_2 * state.RhoList(last) * 
                             Rl_3_minus_Rl_1_3_inv;
        
        Hydrob(last) = -4.0 * Rl_2 * 
                      (state.pList(last+1) - state.pList(last)) - 
                      state.MhyList(last) * (state.RhoList(last) + 
                      state.RhoList(last+1)) * 
                      (Rl1 - Rl_1);
    }
    
    void calculateDeltaRhoAndP() {
        int NoLayers = state.NoLayers;
        
        double R0 = state.RList(0);
        double R0_2 = R0 * R0;
        double R0_3 = R0_2 * R0;
        
        deltaRho(0) = -3.0 * state.RhoList(0) * R0_2 * 
                     deltaR(0) / R0_3;
        deltap(0) = -5.0 * state.pList(0) * R0_2 * 
                   deltaR(0) / R0_3;
        
        for (int i = 1; i < (NoLayers-1); i++) {
            double Ri = state.RList(i);
            double Ri_2 = Ri * Ri;
            double Ri_3 = Ri_2 * Ri;
            double Ri_1 = state.RList(i-1);
            double Ri_1_2 = Ri_1 * Ri_1;
            double Ri_1_3 = Ri_1_2 * Ri_1;
            double Ri_Ri_1 = Ri - Ri_1;
            double Ri_Ri_1_sum_of_squares = Ri_2 + Ri*Ri_1 + Ri_1_2;
            double Ri_3_minus_Ri_1_3_inv = 1.0 / (Ri_Ri_1 * Ri_Ri_1_sum_of_squares);
            
            deltaRho(i) = -3.0 * state.RhoList(i) * (Ri_2 * deltaR(i) - 
                         Ri_1_2 * deltaR(i-1)) * 
                         Ri_3_minus_Ri_1_3_inv;
            deltap(i) = -5.0 * state.pList(i) * (Ri_2 * deltaR(i) - 
                       Ri_1_2 * deltaR(i-1)) * 
                       Ri_3_minus_Ri_1_3_inv;
        }
    }
    
    void updateProfiles() {
        int NoLayers = state.NoLayers;
        
        // Update enclosed total mass after relaxation
        for (int i = 0; i < (NoLayers-1); i++) {
            state.MhyList(i) = state.MList(i) + MbaryonP(state.RList(i), params.mass_norm, params.scale_norm);
        }
        
        // Update u after relaxation
        state.uList = (1.5) * state.aList * state.RhoList.pow(2.0/3.0);

        // Update v after relaxation
        state.vList = std::sqrt(2.0/3.0) * state.uList.sqrt();
        
        // Update L profile
        double R0 = state.RList(0);
        double R0_2 = R0 * R0;
        double sigma_2 = params.sigma * params.sigma;
        double v0 = state.vList(0);
        double v0_2 = v0 * v0;
        double v0_3 = v0_2 * v0;
        double v1 = state.vList(1);
        double v1_2 = v1 * v1;
        double v1_3 = v1_2 * v1;
        
        state.LList(0) = -(state.uList(1) - state.uList(0)) / state.RList(1) * 
                       R0_2 * params.a * params.b * params.c * params.sigma *
                       (state.RhoList(0) * v0_3 / 
                       (params.a * params.c * sigma_2 * state.RhoList(0) * 
                       v0_2 + params.b) + 
                       state.RhoList(1) * v1_3 / 
                       (params.a * params.c * sigma_2 * state.RhoList(1) * 
                       v1_2 + params.b));
        
        for (int i = 1; i < (NoLayers-1); i++) {
            double Ri = state.RList(i);
            double Ri_2 = Ri * Ri;
            double Ri1 = state.RList(i+1);
            double Ri_1 = state.RList(i-1);
            
            double vi = state.vList(i);
            double vi_2 = vi * vi;
            double vi_3 = vi_2 * vi;
            
            double vi1 = state.vList(i+1);
            double vi1_2 = vi1 * vi1;
            double vi1_3 = vi1_2 * vi1;
            
            state.LList(i) = -(state.uList(i+1) - state.uList(i)) / 
                           (Ri1 - Ri_1) * 
                           Ri_2 * params.a * params.b * params.c * params.sigma *
                           (state.RhoList(i) * vi_3 / 
                           (params.a * params.c * sigma_2 * state.RhoList(i) * 
                           vi_2 + params.b) + 
                           state.RhoList(i+1) * vi1_3 / 
                           (params.a * params.c * sigma_2 * state.RhoList(i+1) * 
                           vi1_2 + params.b));
        }
    }
    
    void saveResults(std::ofstream& file, int tstep) {
        if (file.is_open()) {
            file << std::scientific << std::setprecision(10) << params.totalTime << " " << tstep << '\n'
                << state.RList.transpose() << '\n'
                << state.RhoList.transpose() << '\n'
                << state.MList.transpose() << '\n'
                << state.uList.transpose() << '\n'
                << state.LList.transpose() << '\n';
        }
    }
};

int main() {
    clock_t start = clock();
    
    try {
        // Create logger
        Logger logger(false);  // Set to true for debug messages
        
        // Create simulation parameters
        SimulationParameters params;
        
        // Create file manager
        FileManager fileManager(logger);
        
        // Display parameters
        params.display(logger);
        
        // Create and initialize simulator
        Simulator simulator(params, fileManager, logger);
        simulator.initialize();
        
        // Run simulation
        simulator.runSimulation();
        
        // Measure computation time
        clock_t end = clock();
        float seconds = (float)(end - start) / CLOCKS_PER_SEC;
        logger.info("Computation time = " + std::to_string(seconds) + " s");
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    
    return EXIT_SUCCESS;
}